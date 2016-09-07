package neuralnet

import (
	"reflect"
	"testing"
)

var multMatrixVectorTests = []struct {
	matIn  [][]float64
	vectIn []float64
	want   []float64
}{
	{
		[][]float64{
			{2, 3, 5},
			{4, 9, 1},
		},
		[]float64{1, 2, 3},
		[]float64{23, 25},
	},
	{
		[][]float64{
			{0, 0},
			{4, 0},
		},
		[]float64{1, 1},
		[]float64{0, 4},
	},
}

func TestMultMatrixVect(t *testing.T) {
	for i, tt := range multMatrixVectorTests {
		got := multMatrixVect(tt.matIn, tt.vectIn)

		if !reflect.DeepEqual(tt.want, got) {
			t.Errorf("test %d:\nmatIn:%v\nvectIn:%v\nwant:%v\ngot:%v",
				i, tt.matIn, tt.vectIn, tt.want, got)
		}
	}
}

var multTransposeMatVectorTest = []struct {
	matIn  [][]float64
	vectIn []float64
	want   []float64
}{
	{
		[][]float64{
			{2, 3, 5},
			{4, 9, 1},
		},
		[]float64{1, 2},
		[]float64{10, 21, 7},
	},

	{
		[][]float64{
			{12, 1, -8},
			{6, -3, 2},
			{9, 0, -4},
		},
		[]float64{0, 4, -1},
		[]float64{15, -12, 12},
	},
}

func TestMultTransposeMatVector(t *testing.T) {
	for i, tt := range multTransposeMatVectorTest {
		got := multTransposeMatVector(tt.matIn, tt.vectIn)

		if !reflect.DeepEqual(tt.want, got) {
			t.Errorf("test %d:\nmatIn:%v\nvectIn:%v\nwant:%v\ngot:%v",
				i, tt.matIn, tt.vectIn, tt.want, got)
		}
	}
}

var hadamardProdVectTests = []struct {
	vectIn1 []float64
	vectIn2 []float64
	want    []float64
}{
	{
		[]float64{2, 3, 5},
		[]float64{1, 2, 12},
		[]float64{2, 6, 60},
	},
	{
		[]float64{-1, 4, 5, 102},
		[]float64{0, 3, 13, -1},
		[]float64{0, 12, 65, -102},
	},
}

func TestHadamardProdVect(t *testing.T) {
	for i, tt := range hadamardProdVectTests {
		got := hadamardProdVect(tt.vectIn1, tt.vectIn2)

		if !reflect.DeepEqual(tt.want, got) {
			t.Errorf("test %d:\nvectIn1:%v\nvectIn2:%v\nwant:%v\ngot:%v",
				i, tt.vectIn1, tt.vectIn2, tt.want, got)
		}
	}
}

var multVectTransposeVectTests = []struct {
	vectIn1 []float64
	vectIn2 []float64
	want    [][]float64
}{
	{
		[]float64{2, 3, 5},
		[]float64{1, 2},
		[][]float64{
			{2, 4},
			{3, 6},
			{5, 10},
		},
	},
	{
		[]float64{-1, 2},
		[]float64{1, 4, 5, 9},
		[][]float64{
			{-1, -4, -5, -9},
			{2, 8, 10, 18},
		},
	},
	{
		[]float64{-1},
		[]float64{1},
		[][]float64{
			{-1},
		},
	},
}

func TestMultVectTransposeVect(t *testing.T) {
	for i, tt := range multVectTransposeVectTests {
		got := multVectTransposeVect(tt.vectIn1, tt.vectIn2)

		if !reflect.DeepEqual(tt.want, got) {
			t.Errorf("test %d:\nvectIn1:%v\nvectIn2:%v\nwant:%v\ngot:%v ",
				i, tt.vectIn1, tt.vectIn2, tt.want, got)
		}
	}
}

var addVectOfVectsTests = []struct {
	matIn1 [][]float64
	matIn2 [][]float64
	want   [][]float64
}{
	{
		[][]float64{
			{2, 4},
			{3, 6},
			{5, 10},
		},
		[][]float64{
			{9, 1},
			{3, 1223},
			{5, -23},
		},
		[][]float64{
			{11, 5},
			{6, 1229},
			{10, -13},
		},
	},
}

func TestAddVectOfVects(t *testing.T) {
	for i, tt := range addVectOfVectsTests {
		got := addVectOfVects(tt.matIn1, tt.matIn2)

		if !reflect.DeepEqual(tt.want, got) {
			t.Errorf("test %d:\nmatIn1:%v\nvmatIn2:%v\nwant:%v\ngot:%v ",
				i, tt.matIn1, tt.matIn2, tt.want, got)
		}
	}
}
