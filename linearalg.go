package neuralnet

func multMatrixVect(m [][]float64, v []float64) []float64 {
	result := make([]float64, len(m))

	for i := range result {
		var sum float64
		for j := range v {
			sum += m[i][j] * v[j]
		}

		result[i] = sum
	}

	return result
}

func applySigmoidVector(v []float64) {
	for i := range v {
		v[i] = Sigmoid(v[i])
	}
}

func sigmoidDiffVector(v []float64) []float64 {
	res := make([]float64, len(v))
	for i, val := range v {
		res[i] = SigmoidDiff(val)
	}

	return res
}

func multTransposeMatVector(m [][]float64, v []float64) []float64 {
	row, col := len(m), len(m[0])

	res := make([]float64, col)

	for i := 0; i < col; i++ {
		sum := float64(0)
		for j := 0; j < row; j++ {
			sum += m[j][i] * v[j]
		}

		res[i] = sum
	}

	return res
}

func hadamardProdVect(v1 []float64, v2 []float64) []float64 {
	res := make([]float64, len(v1))

	for i := range v1 {
		res[i] = v1[i] * v2[i]
	}

	return res
}

func multVectTransposeVect(v1 []float64, v2 []float64) [][]float64 {
	res := make([][]float64, len(v1))

	for i := range v1 {
		res[i] = make([]float64, len(v2))

		for j := range v2 {
			res[i][j] = v1[i] * v2[j]
		}
	}

	return res
}

// subVects returns v1-v2.
// Panics if the vector sizes don't match.
func subVects(v1, v2 []float64) []float64 {
	if len(v1) != len(v2) {
		panic("neuralnet - subVects: vector sizes don't match")
	}

	res := make([]float64, len(v1))
	for i := range res {
		res[i] = v1[i] - v2[i]
	}

	return res
}

// addVects returns v1+v2.
// Panics if sizes don't match.
func addVects(v1, v2 []float64) []float64 {
	if len(v1) != len(v2) {
		panic("neuralnet - addVects: vector sizes don't match")
	}

	res := make([]float64, len(v1))
	for i := range res {
		res[i] = v1[i] + v2[i]
	}

	return res
}

// sigmoidVect return sigmoid(v).
func sigmoidVect(v []float64) []float64 {
	res := make([]float64, len(v))

	for i, val := range v {
		res[i] = Sigmoid(val)
	}

	return res
}

// multScalarVect return x*v.
func multScalarVect(x float64, v []float64) []float64 {
	res := make([]float64, len(v))

	for i, val := range v {
		res[i] = x * val
	}

	return res
}

// addVectOfVects adds vectors in another vector and returns the sum.
func addVectOfVects(x, y [][]float64) [][]float64 {
	if len(x) != len(y) {
		panic("neuralnet: addVectOfVects: lenghs don't match")
	}

	res := make([][]float64, len(x))
	for i := range x {
		if len(x[i]) != len(y[i]) {
			panic("neuralnet: addVectOfVects: lenghs don't match")
		}

		res[i] = make([]float64, len(x[i]))
		for j := range x[i] {
			res[i][j] = x[i][j] + y[i][j]
		}
	}

	return res
}

// addVectOfMats adds the matrices presents in both vectors and returns the sum.
func addVectOfMats(v1, v2 [][][]float64) [][][]float64 {
	if len(v1) != len(v2) {
		panic("neuralnet: addVectOfMats: lenghs don't match" + string(len(v1)) +
			"!=" + string(len(v2)))
	}

	res := make([][][]float64, len(v1))
	for i := range v1 {
		res[i] = addVectOfVects(v1[i], v2[i])
	}

	return res
}
