package org.opencv.samples.imagemanipulations;

/**
 * Created by Jcama on 29/12/2015.
 */
public class MyThread implements Runnable {

    public MyThread(Object parameter) {
        // store parameter for later user
    }

    public void run() {
        System.out.println("Hola Thread!");
    }
}
