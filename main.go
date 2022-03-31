package main

import (
	"bytes"
	"fmt"
	pigo "github.com/esimov/pigo/core"
	"github.com/fogleman/gg"
	"image"
	"image/color"
	"image/jpeg"
	"io"
	"io/ioutil"
	"log"
	"math"
	"os"
)

func main() {

	input := "input/1.jpeg"
	out := "out/1_out.jpeg"

	cascadeFile, err := ioutil.ReadFile("cascade/facefinder")
	if err != nil {
		log.Fatalf("Error reading the cascade file: %v", err)
	}

	src, err := pigo.GetImage(input)
	if err != nil {
		log.Fatalf("Cannot open the image file: %v", err)
	}

	pixels := pigo.RgbToGrayscale(src)
	cols, rows := src.Bounds().Max.X, src.Bounds().Max.Y

	cParams := pigo.CascadeParams{
		MinSize:     20,
		MaxSize:     1000,
		ShiftFactor: 0.1,
		ScaleFactor: 1.1,

		ImageParams: pigo.ImageParams{
			Pixels: pixels,
			Rows:   rows,
			Cols:   cols,
			Dim:    cols,
		},
	}

	pigo := pigo.NewPigo()
	// Unpack the binary file. This will return the number of cascade trees,
	// the tree depth, the threshold and the prediction from tree's leaf nodes.
	classifier, err := pigo.Unpack(cascadeFile)
	if err != nil {
		log.Fatalf("Error reading the cascade file: %s", err)
	}

	angle := 0.0 // cascade rotation angle. 0.0 is 0 radians and 1.0 is 2*pi radians

	// Run the classifier over the obtained leaf nodes and return the detection results.
	// The result contains quadruplets representing the row, column, scale and detection score.
	dets := classifier.RunCascade(cParams, angle)

	// Calculate the intersection over union (IoU) of two clusters.
	dets = classifier.ClusterDetections(dets, 0.1)
	fmt.Println(dets)

	dc = gg.NewContext(cols, rows)
	dc.DrawImage(src, 0, 0)

	buff := new(bytes.Buffer)
	drawMarker(dets, buff, false)

	// buffer to image
	img, _, err := image.Decode(buff)
	if err != nil {
		panic(err)
	}
	f, _ := os.Create(out)   //创建文件
	defer f.Close()          //关闭文件
	jpeg.Encode(f, img, nil) //写入文件
}

var dc *gg.Context

// drawMarker mark the detected face region with the provided
// marker (rectangle or circle) and write it to io.Writer.
func drawMarker(detections []pigo.Detection, w io.Writer, isCircle bool) error {
	var qThresh float32 = 5.0

	for i := 0; i < len(detections); i++ {
		if detections[i].Q > qThresh {
			if isCircle {
				dc.DrawArc(
					float64(detections[i].Col),
					float64(detections[i].Row),
					float64(detections[i].Scale/2),
					0,
					2*math.Pi,
				)
			} else {
				dc.DrawRectangle(
					float64(detections[i].Col-detections[i].Scale/2),
					float64(detections[i].Row-detections[i].Scale/2),
					float64(detections[i].Scale),
					float64(detections[i].Scale),
				)
			}
			dc.SetLineWidth(3.0)
			dc.SetStrokeStyle(gg.NewSolidPattern(color.RGBA{R: 255, G: 0, B: 0, A: 255}))
			dc.Stroke()
		}
	}
	return dc.EncodePNG(w)
}
