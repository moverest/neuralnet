package neuralnet

import (
	"fmt"
	"math"
	"testing"
)

var sdgTests = []struct {
	net          Network
	errors       [][]float64
	activations  [][]float64
	want         []float64
	weighted     [][]float64
	deltas       [][]float64
	weightDeltas [][][]float64
	biasDeltas   [][]float64
	learningRate float64
	floatErr     float64 // float precision being used to compare
}{
	{
		net: Network{
			Sizes: []int{2, 3, 2},

			Weights: [][][]float64{
				{
					{1, 2},
					{-3, 6},
					{7, -4},
				},
				{
					{-6, 3, -1},
					{4, 2, -12},
				},
			},

			Biases: [][]float64{
				{6, 4, 2},
				{2, 5},
			},
		},

		want: []float64{1, 0},

		activations: [][]float64{
			{1, 0},
			{0.99908895, 0.73105858, 0.99987661},
			{0.0572575, 0.17652854},
		},

		weighted: [][]float64{
			{7, 1, 9},
			{-2.80123456, -1.54004631},
		},

		deltas: [][]float64{
			{3.71347822e-04, -1.99251527e-02, -3.17142996e-05},
			{-0.05088837, 0.02566129},
		},

		weightDeltas: [][][]float64{
			{
				{-0.00148539, 0.},
				{0.07970061, 0.},
				{0.00012686, 0.},
			},
			{
				{0.20336804, 0.14880952, 0.20352837},
				{-0.10255163, -0.07503961, -0.10263248},
			},
		},

		biasDeltas: [][]float64{
			{-0.00148539, 0.07970061, 0.00012686},
			{0.20355348, -0.10264514},
		},

		learningRate: 4,
		floatErr:     1e-6,
	},
}

// floatEqual compares two [][][]float64 [][]float64 []float64 float64 and
// return false if the differences ain't below epsilon.
func floatEqual(epsilon float64, x, y interface{}) bool {
	// Should be rewritten someday... but it works for what we want here.

	switch v := x.(type) {
	case float64:
		w, ok := y.(float64)
		if !ok {
			return false
		}

		return math.Abs(v-w) <= epsilon

	case []float64:
		w, ok := y.([]float64)
		if !ok {
			return false
		}

		if len(v) != len(w) {
			return false
		}

		for i := 0; i < len(v); i++ {
			if !floatEqual(epsilon, v[i], w[i]) {
				return false
			}
		}
		return true

	case [][]float64:
		w, ok := y.([][]float64)
		if !ok {
			return false
		}

		if len(v) != len(w) {
			return false
		}

		for i := 0; i < len(v); i++ {
			if !floatEqual(epsilon, v[i], w[i]) {
				return false
			}
		}
		return true

	case [][][]float64:
		w, ok := y.([][][]float64)
		if !ok {
			return false
		}

		if len(v) != len(w) {
			return false
		}

		for i := 0; i < len(v); i++ {
			if !floatEqual(epsilon, v[i], w[i]) {
				return false
			}
		}
		return true

	default:
		fmt.Println("default")
		return false
	}
}

var floatEqualTests = []struct {
	xIn, yIn interface{}
	epsIn    float64
	want     bool
}{
	{
		float64(2),
		float64(2),
		1e-20,
		true,
	},
	{
		[]float64{1, 4, 1},
		[]float64{1, 4},
		1e-10,
		false,
	},
	{
		[]float64{1, 4, 6},
		[]float64{1, 4, 6},
		1e-10,
		true,
	},
	{
		[]float64{1, 4, 6},
		[]float64{1, 3, 6},
		1e-10,
		false,
	},
	{
		[][]float64{{1, 4, 6}, {4}},
		[][]float64{{1, 4, 6}, {4}},
		1e-10,
		true,
	},
	{
		[][]float64{{1, 4, 6}, {4}},
		[]float64{1, 4, 6},
		1e-10,
		false,
	},
	{
		[][]float64{{7, 1, 9}, {-2.80123456, -1.54004631}},
		[][]float64{{7, 1, 9}, {-2.801234562367595, -1.5400463126057575}},
		1e-6,
		true,
	},
}

func TestFloatEqual(t *testing.T) {
	for i, tt := range floatEqualTests {
		if floatEqual(tt.epsIn, tt.xIn, tt.yIn) != tt.want {
			t.Errorf("test %d:\ninEps:%v\ninX:%v\ninY:%v\nwant:%v\ngot:%v",
				i, tt.epsIn, tt.xIn, tt.yIn, tt.want, !tt.want)
		}
	}
}

func TestSgd(t *testing.T) {
	for i, tt := range sdgTests {
		activations, weighted := tt.net.sdgComputeAZ(tt.activations[0])

		if !floatEqual(tt.floatErr, activations, tt.activations) {
			t.Errorf("test %d:\nactivations:%v\ngot:%v", i, tt.activations, activations)
		}

		if !floatEqual(tt.floatErr, weighted, tt.weighted) {
			t.Errorf("test %d:\nweighted:%v\ngot:%v", i, tt.weighted, weighted)
		}

		deltas := tt.net.sdgComputeErr(activations, weighted, tt.want)

		if !floatEqual(tt.floatErr, deltas, tt.deltas) {
			t.Errorf("test %d:\nweighted:%v\ngot:%v", i, tt.deltas, deltas)
		}

		weightDeltas, biasDeltas := tt.net.sdgComputeWBDiffs(deltas, activations,
			tt.learningRate)

		if !floatEqual(tt.floatErr, weightDeltas, tt.weightDeltas) {
			t.Errorf("test %d:\nweighted:%v\ngot:%v", i, tt.weightDeltas, weightDeltas)
		}

		if !floatEqual(tt.floatErr, biasDeltas, tt.biasDeltas) {
			t.Errorf("test %d:\nweighted:%v\ngot:%v", i, tt.biasDeltas, biasDeltas)
		}

	}
}
