package src

import (
	"math/rand"
	"bytes"
	"strconv"
)

type Rect struct {
	Width int;
	Height int;
	Data []float64
}

func NewRect(x, y int) *Rect {
	ret := &Rect{Width: x, Height:y, Data: make([]float64, x * y, x * y)}
	return ret;
}

func NewRectWithData(x, y int, data []float64) *Rect {
	return &Rect{Width:x, Height:y, Data: data}
}

func (rect *Rect)Randomize() *Rect{
	for i := (rect.Width * rect.Height - 1); i >= 0; i -- {
		rect.Data[i] = rand.Float64() - 0.5;
	}
	return rect
}

func (rect *Rect)Get(x, y int) float64 {
	if x < 0 || x >= rect.Width || y < 0 || y >= rect.Height {
		panic("out of bounds " + string(x) + "," + string(y));
	}else{
		return rect.Data[y * rect.Width + x];
	}
}

func (rect *Rect)Set(x, y int, data float64) {
	if x < 0 || x >= rect.Width || y < 0 || y >= rect.Height {
		panic("out of bounds " + string(x) + "," + string(y));
	}else{
		rect.Data[y * rect.Width + x] = data;
	}
}

func (rect *Rect)Junc(other *Rect) *Rect {
	if(rect.Height != other.Width){
		panic("junc requires that the height of the first rect same with the width of the second: first Height " + strconv.Itoa(rect.Height) + " second Width " + strconv.Itoa(other.Width));
	}

	len := rect.Width * other.Height;
	data := make([]float64, len, len);
	for w := 0 ; w < rect.Width; w ++ {
		for h := 0; h < other.Height; h ++ {
			index := h * rect.Width + w;
			for k := 0; k < rect.Height; k ++ {
				data[index] += rect.Get(w, k) * other.Get(k, h);
			}
		}
	}

	return NewRectWithData(rect.Width, other.Height, data)
}

type Handler func(data float64, x , y int) float64;

func (rect *Rect)Operate(oper Handler) *Rect{
	for x := 0; x < rect.Width; x ++ {
		for y := 0; y < rect.Height; y ++ {
			rect.Set(x, y, oper(rect.Get(x, y), x, y));
		}
	}
	return rect;
}

func OperateWrapper(oper func(float64) float64) Handler{
	return func(data float64, x, y int) float64{
		return oper(data);
	}
}

func (rect *Rect)Add(other *Rect) *Rect {
	if(rect.Width != other.Width || rect.Height != other.Height){
		panic("add require rect of the same size");
	}else{
		return rect.Operate(func(data float64, x, y int) float64{
			return data + other.Get(x, y);
		});
	}
}

func (rect *Rect)Minus(other *Rect) *Rect {
	if(rect.Width != other.Width || rect.Height != other.Height){
		panic("add require rect of the same size");
	}else{
		return rect.Operate(func(data float64, x, y int) float64{
			return data - other.Get(x, y);
		});
	}
}

func (rect *Rect)Transpose() *Rect {
	ret := NewRect(rect.Height, rect.Width)
	return ret.Operate(func(data float64, x, y int) float64{
		return rect.Get(y, x);
	});
}

func (rect Rect)String() string {
	var buffer bytes.Buffer
	for h := 0; h < rect.Height; h ++ {
		for w := 0; w < rect.Width; w ++ {
			buffer.WriteString(strconv.FormatFloat(rect.Get(w, h), 'f', -1, 64))
			buffer.WriteString(",")
		}
		buffer.WriteString("\n")
	}
	return buffer.String()
}

func (rect *Rect)Copy() *Rect {
	return NewRect(rect.Width, rect.Height).Operate(func(data float64, x, y int) float64{
		return rect.Get(x, y);
	});
}