package neuralnet

import (
	"log"
	"math/rand"
)

// SGD trains the network with de stochastic gradient descent algorithm.
//
// If no test set is given (by passing nil), SGD will not evaluate the network
// performance on every epoch.
// The evaluate function is used to evaluate the network. It should return the
// number of correct guesses.
func (net *Network) SGD(training, test Set, evaluate func(*Network, Set) int,
	epochNum, epochSize int, learningRate float64) {

	for i := 0; i < epochNum; i++ {
		suffledSet := rand.Perm(training.Count())

		for j := 0; j < len(suffledSet)/epochSize; j++ {
			// Well the last one may be cut out... maybe it will be fixed someday.
			net.sgdEpoch(training, suffledSet[j*epochSize:(j+1)*(epochSize)], learningRate)
		}

		log.Printf("Epoch %d: %d/%d\n", i, evaluate(net, test), test.Count())
	}
}

// sgdEpoch select a random sample of n items from the set and apply the SDG
// algorithm on it.
// Panics if epoch is empty.
func (net *Network) sgdEpoch(training Set, epoch []int, learningRate float64) {
	if len(epoch) == 0 {
		panic("empty epoch")
	}

	var weightDeltas [][][]float64
	var biasDeltas [][]float64

	for i := 0; i < len(epoch); i++ {
		in, expectedOut := training.GetVects(epoch[i])
		wD, bD := net.sgdComputeWBDeltas(in, expectedOut, learningRate/float64(len(epoch)))

		if weightDeltas == nil {
			weightDeltas = wD
			biasDeltas = bD
		} else {
			weightDeltas = addVectOfMats(weightDeltas, wD)
			biasDeltas = addVectOfVects(biasDeltas, bD)
		}
	}

	net.Weights = addVectOfMats(net.Weights, weightDeltas)
	net.Biases = addVectOfVects(net.Biases, biasDeltas)
}

// sgdComputeWBDeltas computes the weight and bias deltas needed to be add to
// the current weights and biases to meet the expected output.
func (net *Network) sgdComputeWBDeltas(in, expectedOut []float64,
	learningRate float64) (weightDeltas [][][]float64, biasDeltas [][]float64) {

	activations, weighted := net.sdgComputeAZ(in)
	deltas := net.sdgComputeErr(activations, weighted, expectedOut)
	weightDeltas, biasDeltas = net.sdgComputeWBDiffs(deltas, activations, learningRate)

	return
}

// sdgComputeAZ computes the neuron's activations and weighted inputs
func (net *Network) sdgComputeAZ(in []float64) (activations, weighted [][]float64) {
	activations = make([][]float64, len(net.Sizes))
	weighted = make([][]float64, len(net.Biases))

	activations[0] = in

	for i := range net.Biases {
		weighted[i] = addVects(multMatrixVect(net.Weights[i], activations[i]),
			net.Biases[i])
		activations[i+1] = sigmoidVect(weighted[i])
	}

	return
}

// sdgComputeErr computes the errors
func (net *Network) sdgComputeErr(activations, weighted [][]float64,
	expectedOut []float64) (deltas [][]float64) {
	n := len(net.Biases)
	deltas = make([][]float64, n)

	deltas[n-1] = hadamardProdVect(
		subVects(activations[n], expectedOut),
		sigmoidDiffVector(weighted[n-1]))

	for i := n - 2; i >= 0; i-- {
		deltas[i] = hadamardProdVect(
			multTransposeMatVector(net.Weights[i+1], deltas[i+1]),
			sigmoidDiffVector(weighted[i]))
	}

	return
}

// sdgComputeWBDiffs computes the weight and bias deltas to be added the the
// current weights and biases to meet the expected ouput given the errors and
// the activations.
func (net *Network) sdgComputeWBDiffs(deltas, activations [][]float64,
	learningRate float64) (weightDeltas [][][]float64, biasDeltas [][]float64) {
	weightDeltas = make([][][]float64, len(net.Weights))
	biasDeltas = make([][]float64, len(net.Biases))

	for i := range net.Biases {
		weightDeltas[i] = multVectTransposeVect(
			multScalarVect(-learningRate, deltas[i]),
			activations[i])

		biasDeltas[i] = multScalarVect(-learningRate, deltas[i])
	}

	return
}
