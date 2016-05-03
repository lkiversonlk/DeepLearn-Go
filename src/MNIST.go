package src

import (
	"io/ioutil"
	"encoding/binary"
	"fmt"

)

type Image *Rect;
type Label *Rect;

type MNISTData struct {
	Image *Rect
	Label *Rect
}

type MNIST struct{
        Data []*MNISTData
	ImageRow int
	ImageHeight int
}

func ParseMNIST(imageFilePath string, labelFilePath string) (*MNIST, error){

	ret := &MNIST{};

	if imageDatas, err := ioutil.ReadFile(imageFilePath); err != nil{
		return nil, err
	} else {
		magicN := binary.BigEndian.Uint32(imageDatas)
		imageCount := binary.BigEndian.Uint32(imageDatas[4:])
		rows := binary.BigEndian.Uint32(imageDatas[8:])
		cols := binary.BigEndian.Uint32(imageDatas[12:])
		fmt.Printf("parsing image file: magic %d, image count %d, rows %d, cols %d\n", magicN, imageCount, rows, cols)

		ret.ImageRow = int(rows);
		ret.ImageHeight = int(cols);

		ret.Data = make([] *MNISTData, imageCount, imageCount)

		imageDataW := rows * cols;
		start := 16
		for i := 0; i < int(imageCount); i ++ {
			imageData := make([]float64, int(imageDataW), int(imageDataW))
			for j := 0; j < int(imageDataW); j++ {
				imageData[j] = (float64(uint(imageDatas[start + j])) / 255.0)
			}
			ret.Data[i] = &MNISTData{Image : NewRectWithData(int(imageDataW), 1, imageData)}
			start += int(imageDataW)
		}
	}

	if labelDatas, err := ioutil.ReadFile(labelFilePath); err != nil{
		return nil, err
	} else {
		magicN := binary.BigEndian.Uint32(labelDatas)
		labelCount := int(binary.BigEndian.Uint32(labelDatas[4:]))
		fmt.Printf("parsing label file: magic %v, label count %v\n", magicN, labelCount)

		if(labelCount != len(ret.Data)){
			panic("MNIST data error, length of iamge not equal to length of label")
		}


		start := 8
		for i := 0; i < int(labelCount); i ++ {
			labelData := make([]float64, 10, 10)
			label := int(labelDatas[start + i])
			labelData[label] = 1
			ret.Data[i].Label = NewRectWithData(10, 1, labelData)
		}
	}

	return ret, nil
}
