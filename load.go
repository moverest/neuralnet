package neuralnet

import (
	"encoding/binary"
	"errors"
	"os"
)

var (
	// ErrFormat indicates that the file format is invalid.
	ErrFormat = errors.New("neuralnet: Invalid format")
)

// LoadFile loads a network from a file, and returns it.
func LoadFile(name string) (*Network, error) {
	// File format (big endian):
	// int32	number of layers (n)
	// int32	layer 0 size
	// ...
	// int32	layer (n-1) size
	// float64	weight
	// ...
	// float64	weight
	// float64	bias
	// ...
	// float64	bias

	file, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Gets the number of layers
	var numLayers int32
	err = binary.Read(file, binary.BigEndian, &numLayers)
	if err != nil {
		return nil, err
	}
	if numLayers < 0 {
		return nil, ErrFormat
	}

	// Loads the sizes
	sizes := make([]int, int(numLayers))
	var size int32
	for i := range sizes {
		err = binary.Read(file, binary.BigEndian, &size)
		if err != nil {
			return nil, err
		}
		if size < 0 {
			return nil, ErrFormat
		}

		sizes[i] = int(size)
	}

	net := New(sizes)

	// Loads the weights
	for i := range net.Weights {
		for j := range net.Weights[i] {
			for k := range net.Weights[i][j] {
				err = binary.Read(file, binary.BigEndian, &net.Weights[i][j][k])
				if err != nil {
					return nil, err
				}
			}
		}
	}

	// Loads the biases
	for i := range net.Biases {
		for j := range net.Biases[i] {
			err = binary.Read(file, binary.BigEndian, &net.Biases[i][j])
			if err != nil {
				return nil, err
			}
		}
	}

	return net, nil
}

// SaveFile saves the network to a file.
func (net *Network) SaveFile(name string) error {
	file, err := os.Create(name)
	if err != nil {
		return err
	}
	defer file.Close()

	// Saves the number of layers
	err = binary.Write(file, binary.BigEndian, int32(len(net.Sizes)))
	if err != nil {
		return err
	}

	// Saves the sizes
	for i := range net.Sizes {
		err = binary.Write(file, binary.BigEndian, int32(net.Sizes[i]))
		if err != nil {
			return err
		}
	}

	// Saves the weights
	for i := range net.Weights {
		for j := range net.Weights[i] {
			for k := range net.Weights[i][j] {
				// float64() is used to make sure saving won't get broken in the case
				// types change internally
				err = binary.Write(file, binary.BigEndian, float64(net.Weights[i][j][k]))
				if err != nil {
					return err
				}
			}
		}
	}

	// Saves the biases
	for i := range net.Biases {
		for j := range net.Biases[i] {
			err = binary.Write(file, binary.BigEndian, float64(net.Biases[i][j]))
			if err != nil {
				return err
			}
		}
	}

	return nil
}
