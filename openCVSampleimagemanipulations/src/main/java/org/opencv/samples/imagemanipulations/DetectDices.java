package org.opencv.samples.imagemanipulations;

import android.app.Activity;
import android.os.AsyncTask;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.io.Console;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Created by Jcama on 29/12/2015.
 */

public class DetectDices extends AsyncTask<Mat, Void, Void> {

    private Mat frame = null;
    private List<MatOfPoint> squares = null;

    public DetectDices() {

    }

    public Void doInBackground(Mat... params) {
        mainDetect(params[0], params[1]);
        return null;
    }

    public Mat GetFrame(){
        return this.frame;
    }

    public List<MatOfPoint> GetSquares(){
        return this.squares;
    }

    public void mainDetect(Mat rgba, Mat gray){
        Size sizeRgba = rgba.size();
        Mat rgbaInnerWindow;
        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;
        Mat cl1 = new Mat();
        rgbaInnerWindow = rgba.submat(0, rows, 0, cols);
        Equalizer(gray, cl1);
        Blur(cl1, rgbaInnerWindow);
        Dilate_Erode(rgbaInnerWindow, cl1);
        this.squares = SquaresFilter(Find_Squares(cl1));
        //this.squares = Find_Squares(cl1);
        this.frame = cl1.clone();
    }

    private void Equalizer(Mat img, Mat cl1) {
        Mat mattemp = new Mat();
        double clipLimit = 1.2;
        Size tileGridSize = new Size(8, 8);
        CLAHE clahe = Imgproc.createCLAHE(clipLimit, tileGridSize);
        clahe.apply(img, mattemp);
        Imgproc.equalizeHist(mattemp, cl1);
    }

    private void Blur(Mat img, Mat blurDst) {
        Size kernel = new Size(5, 5);
        Imgproc.blur(img, blurDst, kernel);
    }

    private void Dilate_Erode(Mat img, Mat erodeDst) {
        Mat dilated_ker = new Mat().ones(1, 1, CvType.CV_32F);
        Mat erorde_ker = new Mat().ones(6, 6, CvType.CV_32F);
        Mat dilateDst = new Mat();
        Imgproc.dilate(img, dilateDst, dilated_ker);
        Imgproc.erode(dilateDst, erodeDst, erorde_ker);
    }

    private List<MatOfPoint> Find_Squares(Mat img) {
        Mat gaussDst = new Mat();
        Size gaussKer = new Size(5, 5);
        Imgproc.GaussianBlur(img, gaussDst, gaussKer, 0);
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        List<MatOfPoint> squares = new ArrayList<MatOfPoint>();

        Mat bin = new Mat();
        List<Mat> listGrays = new ArrayList<Mat>();
        Core.split(gaussDst, listGrays);
        for (Mat gray : listGrays) {
            for (int thrs = 0; thrs <= 255; thrs += 26) {
                if (thrs == 0) {
                    Mat cannyDst = new Mat();
                    int apertureSize = 5;
                    Imgproc.Canny(img, cannyDst, 0, 50);
                    Mat dilated_ker = new Mat().ones(1, 1, CvType.CV_32F);
                    Imgproc.dilate(cannyDst, bin, dilated_ker);
                } else {
                    double retval = Imgproc.threshold(gray, bin, thrs, 255, Imgproc.THRESH_BINARY);
                }

                Mat hierarchy = new Mat();
                Imgproc.findContours(bin, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
                for (MatOfPoint cnt : contours) {
                    MatOfPoint2f cnt2f = new MatOfPoint2f(cnt.toArray());
                    double cnt_len = Imgproc.arcLength(cnt2f, true);
                    Imgproc.approxPolyDP(cnt2f, cnt2f, 0.02 * cnt_len, true);
                    cnt2f.convertTo(cnt, CvType.CV_32S);
                    if ((cnt.height() == 4) && Imgproc.contourArea(cnt2f) > 1000 &&
                            Imgproc.contourArea(cnt2f) < 100000 && Imgproc.isContourConvex(cnt)) {

                        double[] cos = new double[4];

                        for (int i = 0; i < 4; i++){
                            double[] p0 = cnt.get(i, 0);
                            double[] p1 = cnt.get((i + 1)%4, 0);
                            double[] p2 = cnt.get((i + 2)%4, 0);
                            cos[i] = angle_cos(p0, p1, p2);
                        }

                        if (getMax(cos) < 0.1){
                            squares.add(cnt);
                        }
                    }
                }
            }
        }

        return squares;
    }

    private double angle_cos(double[] p0, double[] p1, double[] p2){
        double[] d1 = new double[2];
        double[] d2 = new double[2];

        for (int i = 0; i < 2; i++) {
            d1[i] = p0[i] - p1[i];
            d2[i] = p2[i] - p1[i];
        }

        double dot = d1[0] * d2[0] + d1[1] * d2[1];
        double dot1 = d1[0] * d1[0] + d1[1] * d1[1];
        double dot2 = d2[0] * d2[0] + d2[1] * d2[1];

        return Math.abs(dot / Math.sqrt(dot1 * dot2));
    }

    private static double getMax(double[] array){
        double max = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
            }
        }

        return max;
    }

    private List<MatOfPoint> SquaresFilter(List<MatOfPoint> squares) {
        List<MatOfPoint> filters = new ArrayList<MatOfPoint>();
        boolean[] sq = new boolean[squares.size()];
        Arrays.fill(sq, true);
        for (int i = 0; i < squares.size(); i++){
            if (sq[i] == true){
                filters.add(squares.get(i));
                sq[i] = false;
            }
            for (int j = 0; j < squares.size(); j++){
                if (sq[j] == true){
                    double[] centerI = Center(squares.get(i)).get(0, 0);
                    double[] centerJ = Center(squares.get(j)).get(0,0);
                    if (Math.abs(centerI[0] - centerJ[0]) < 20 &&
                            Math.abs(centerI[1] - centerJ[1]) < 20){
                        sq[j] = false;
                        break;
                    }
                }
            }
        }

        return filters;
    }

    private MatOfPoint Center(MatOfPoint square){
        Point mid = new Point();
        mid.x = (square.get(0,0)[0] + square.get(2,0)[0]) / 2;
        mid.y = (square.get(0,0)[1] + square.get(2,0)[1]) / 2;
        return new MatOfPoint(mid);
    }

    private void DetectFeatures(List<MatOfPoint> squares){

        for (MatOfPoint square : squares){

        }
    }

    private void CleanBackground(Mat img, MatOfPoint square, Mat res){
        Mat mask = new Mat().zeros(img.rows(), img.cols(), CvType.CV_8U);
        List<MatOfPoint> quads = new ArrayList<MatOfPoint>();
        quads.add(square);
        Imgproc.drawContours(img, quads, -1, new Scalar(255, 255, 255), -1);
        Core.bitwise_and(img, mask, res);
    }
}

