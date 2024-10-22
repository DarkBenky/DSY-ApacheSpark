package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"
)

// Record represents a row of data
type Record struct {
	ID    int
	Value float64
}

// GenerateRandomDataset creates a random dataset of records
func GenerateRandomDataset(size int) []Record {
	rand.Seed(time.Now().UnixNano())
	records := make([]Record, size)
	for i := range records {
		records[i] = Record{
			ID:    i,
			Value: rand.Float64() * 100, // Random float between 0 and 100
		}
	}
	return records
}

// WriteDatasetToCSV writes the dataset to a CSV file
func WriteDatasetToCSV(filename string, dataset []Record) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	for _, record := range dataset {
		err := writer.Write([]string{strconv.Itoa(record.ID), fmt.Sprintf("%f", record.Value)})
		if err != nil {
			return err
		}
	}
	return nil
}

// LoadDatasetFromCSV reads the dataset from a CSV file
func LoadDatasetFromCSV(filename string) ([]Record, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var dataset []Record
	for _, row := range rows {
		id, _ := strconv.Atoi(row[0])
		value, _ := strconv.ParseFloat(row[1], 64)
		dataset = append(dataset, Record{ID: id, Value: value})
	}

	return dataset, nil
}

// FilterData filters records with value greater than threshold
func FilterData(dataset []Record, threshold float64) []Record {
	var filtered []Record
	for _, record := range dataset {
		if record.Value > threshold {
			filtered = append(filtered, record)
		}
	}
	return filtered
}

// TransformData multiplies the value by a constant factor
func TransformData(dataset []Record, factor float64) []Record {
	for i := range dataset {
		dataset[i].Value *= factor
	}
	return dataset
}

// AggregateData sums the values in the dataset
func AggregateData(dataset []Record) float64 {
	var sum float64
	for _, record := range dataset {
		sum += record.Value
	}
	return sum
}

// Spark-style memory pipeline using goroutines for parallel processing
func SparkStyleMemoryPipeline(dataset []Record, chunkSize int, threshold float64, factor float64) float64 {
	// Split the dataset into chunks and process each chunk in parallel
	numChunks := (len(dataset) + chunkSize - 1) / chunkSize
	chunks := make([][]Record, numChunks)
	for i := 0; i < numChunks; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if end > len(dataset) {
			end = len(dataset)
		}
		chunks[i] = dataset[start:end]
	}

	var wg sync.WaitGroup
	results := make([]float64, numChunks)
	for i, chunk := range chunks {
		wg.Add(1)
		go func(i int, chunk []Record) {
			defer wg.Done()
			// Step 2: Filter
			chunk = FilterData(chunk, threshold)
			// Step 3: Transform
			chunk = TransformData(chunk, factor)
			// Step 4: Aggregate
			results[i] = AggregateData(chunk)
		}(i, chunk)
	}

	wg.Wait()

	// Final aggregation (Reduce step)
	total := 0.0
	for _, result := range results {
		total += result
	}

	return total
}

// MapReduce-style disk pipeline (write intermediate results to disk)
func MapReduceDiskPipeline(filename string, chunkSize int, threshold float64, factor float64) float64 {
	// Load the dataset from CSV
	dataset, err := LoadDatasetFromCSV(filename)
	if err != nil {
		log.Fatalf("Failed to load dataset from CSV: %v", err)
	}

	numChunks := (len(dataset) + chunkSize - 1) / chunkSize
	tempFiles := make([]string, numChunks)
	var wg sync.WaitGroup
	mutex := &sync.Mutex{}

	// Map phase: Process each chunk in parallel
	for i := 0; i < numChunks; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if end > len(dataset) {
			end = len(dataset)
		}
		chunk := dataset[start:end]

		wg.Add(1)
		go func(i int, chunk []Record) {
			defer wg.Done()

			// Step 2: Filter
			chunk = FilterData(chunk, threshold)

			// Write filtered data to disk (Map step)
			tempFilename := fmt.Sprintf("temp_%d.csv", i)
			mutex.Lock()
			tempFiles[i] = tempFilename
			mutex.Unlock()
			err = WriteDatasetToCSV(tempFilename, chunk)
			if err != nil {
				log.Fatalf("Failed to write temp file: %v", err)
			}
		}(i, chunk)
	}

	wg.Wait() // Wait for all map operations to finish

	var total float64
	wg.Add(numChunks) // Prepare for reduce phase

	// Reduce phase: Process each chunk in parallel
	for _, tempFile := range tempFiles {
		go func(tempFile string) {
			defer wg.Done()

			// Load each chunk (Reduce step)
			chunk, err := LoadDatasetFromCSV(tempFile)
			if err != nil {
				log.Fatalf("Failed to load temp file: %v", err)
			}

			// Step 3: Transform
			chunk = TransformData(chunk, factor)

			// Step 4: Aggregate
			partialSum := AggregateData(chunk)

			mutex.Lock()
			total += partialSum
			mutex.Unlock()
		}(tempFile)
	}

	wg.Wait() // Wait for all reduce operations to finish

	return total
}

func main() {
	// Generate a random dataset
	dataset := GenerateRandomDataset(1_000_000)

	// Write the dataset to a CSV file
	csvFilename := "data.csv"
	err := WriteDatasetToCSV(csvFilename, dataset)
	if err != nil {
		log.Fatalf("Failed to write dataset to CSV: %v", err)
	}

	chunkSize := 10_000
	threshold := 50.0
	factor := 2.0

	// Spark-style Memory Pipeline
	fmt.Println("Spark-style Memory Pipeline")
	start := time.Now()
	memorySum := SparkStyleMemoryPipeline(dataset, chunkSize, threshold, factor)
	fmt.Printf("Memory pipeline sum: %f\n", memorySum)
	fmt.Printf("Memory pipeline took %s\n\n", time.Since(start))

	// MapReduce-style Disk Pipeline
	fmt.Println("MapReduce-style Disk Pipeline")
	start = time.Now()
	diskSum := MapReduceDiskPipeline(csvFilename, chunkSize, threshold, factor)
	fmt.Printf("Disk pipeline sum: %f\n", diskSum)
	fmt.Printf("Disk pipeline took %s\n", time.Since(start))
}
