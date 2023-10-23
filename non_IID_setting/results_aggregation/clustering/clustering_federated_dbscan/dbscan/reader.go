package dbscan

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

// Constants for reading values.
const (
	DATAPATH = "data"
	ID       = 1
	LR       = 2
	MOM      = 3
	BATCH    = 4
	LOSS     = 5
	ACC      = 6
)

// GetClientsData reads values from an experiment and creates the client(s) objects.
func GetClientsData(datasetName, skewType, skew string, clientNum int) []*Client {
	experiment := fmt.Sprintf("%s/%s_%s_skew_%s_%dclients", DATAPATH, datasetName, skewType, skew, clientNum)

	if datasetName == "mnist" && skewType == "qty" {
		if skew == "0.1" && clientNum == 20 {
			clientNum = 16
		}
		if skew == "0.4" && clientNum == 10 {
			clientNum = 8
		}
		if (skew == "0.4" || skew == "2.0") && clientNum == 20 {
			clientNum = 18
		}
		if (skew == "1.0" || skew == "2.0") && clientNum == 10 {
			clientNum = 9
		}
		if skew == "1.0" && clientNum == 20 {
			clientNum = 17
		}
	}
	clients := make([]*Client, clientNum)

	for i := 0; i < clientNum; i++ {
		filename := fmt.Sprintf("%s/client_%d.csv", experiment, i)
		file, err := os.Open(filename)
		if err != nil {
			panic(err)
		}
		defer file.Close()

		reader := csv.NewReader(file)
		records, err := reader.ReadAll()
		if err != nil {
			panic(err)
		}

		client := &Client{
			ID:   i,
			data: make([]Parameters, len(records)-1),
		}

		for j, record := range records[1:] {
			lr, err := strconv.ParseFloat(record[LR], 64)
			if err != nil {
				panic(err)
			}
			mom, err := strconv.ParseFloat(record[MOM], 64)
			if err != nil {
				panic(err)
			}
			batch, err := strconv.Atoi(record[BATCH])
			if err != nil {
				panic(err)
			}
			loss, err := strconv.ParseFloat(record[LOSS], 64)
			if err != nil {
				panic(err)
			}
			accuracy, err := strconv.ParseFloat(record[ACC], 64)
			if err != nil {
				panic(err)
			}

			client.data[j] = Parameters{lr, mom, float64(batch), loss, accuracy}
		}

		clients[i] = client
	}

	return clients
}
