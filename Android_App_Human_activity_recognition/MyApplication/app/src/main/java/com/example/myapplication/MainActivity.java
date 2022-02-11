package com.example.myapplication;
import java.lang.Math;

import android.app.Activity;
import android.hardware.SensorEventListener;
import android.os.Bundle;
import android.widget.TextView;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorManager;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import com.example.myapplication.ml.TfliteModel;


public class MainActivity extends Activity implements SensorEventListener {
    private SensorManager mSensorManager;

    private TextView acctv, gyrtv, tv;
    private final int n_SAMPLES = 250;

    private static List<Float> Accx;
    private static List<Float> Accy;
    private static List<Float> Accz;
    private static List<Float> Gyrox;
    private static List<Float> Gyroy;
    private static List<Float> Gyroz;
    private int counter = 0;
    List<String> Classes = Arrays.asList( "STANDING","WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "WALKING",  "LAYING", "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE", "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND");


    private static ByteBuffer byteBuffer;

    private Sensor mAcc;
    private Sensor mGyr;


    private TfliteModel model;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        tv=findViewById(R.id.textView);
        acctv=findViewById(R.id.textView_acc);
        gyrtv=findViewById(R.id.textView_gyr);

        Accx = new ArrayList<>();
        Accy = new ArrayList<>();
        Accz = new ArrayList<>();
        Gyrox = new ArrayList<>();
        Gyroy = new ArrayList<>();
        Gyroz = new ArrayList<>();


        mSensorManager = (SensorManager)getSystemService(SENSOR_SERVICE);
        mAcc = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mGyr = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);


        byteBuffer=ByteBuffer.allocateDirect(n_SAMPLES*6*4);

    }



    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        String placeholder;
        switch (sensorEvent.sensor.getType()) {
            case Sensor.TYPE_ACCELEROMETER://
                if (Accx.size()<n_SAMPLES) {
                    Accx.add(sensorEvent.values[0]);
                    Accy.add(sensorEvent.values[1]);
                    Accz.add(sensorEvent.values[2]);
                    placeholder = "Acc:" + round(sensorEvent.values[0]) + "_" + round(sensorEvent.values[1]) + "_" + round(sensorEvent.values[2]) + " ";
                    acctv.setText(placeholder);
                }
                break;
            case Sensor.TYPE_GYROSCOPE://
                if (Gyrox.size()<n_SAMPLES) {
                    Gyrox.add(sensorEvent.values[0]);
                    Gyroy.add(sensorEvent.values[1]);
                    Gyroz.add(sensorEvent.values[2]);
                    placeholder = "Gyro:" + round(sensorEvent.values[0]) + "_" + round(sensorEvent.values[1]) + "_" + round(sensorEvent.values[2]) + " ";
                    gyrtv.setText(placeholder);
                }
                break;
            default:
                break;
        }

       activityPrediction();

    }


    @Override
    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(this, mAcc, 10000);
        mSensorManager.registerListener(this, mGyr, 10000);
    }

    @Override
    protected void onPause() {
        super.onPause();
        mSensorManager.unregisterListener(this);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    private void activityPrediction()
    {  try {
        model = TfliteModel.newInstance(this);
    } catch (IOException e) {
        e.printStackTrace();
    }

        if(Accx.size() == n_SAMPLES && Gyrox.size() == n_SAMPLES) {
            // Mean normalize the signal
            normalize(Accx);
            normalize(Accy);
            normalize(Accz);
            normalize(Gyrox);
            normalize(Gyroy);
            normalize(Gyroz);

            List<Float> input_signal = new ArrayList<>();

            // Copy all x, y and z values to input_signal
            int i;
            input_signal.addAll(Accx);
            input_signal.addAll(Accy);
            input_signal.addAll(Accz);
            input_signal.addAll(Gyrox);
            input_signal.addAll(Gyroy);
            input_signal.addAll(Gyroz);


            // Perform inference using Tensorflow

            for(Float data : input_signal) {
                    byteBuffer.putFloat(data);
                }
                // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, n_SAMPLES, 6}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            TfliteModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] output_array = outputFeature0.getFloatArray();

            //show output
            String placeholder="\n";
            for (i=0; i< Classes.size(); i++){
                if (!Double.isNaN(output_array[i])){
                placeholder+=Classes.get(i)+": "+round(output_array[i]*100f) +"%\n";
            }
                else{placeholder="Waiting ...";}
            }
            tv.setText(placeholder);
            // Clear all the values
            Accx.clear();Accy.clear();Accz.clear();
            Gyrox.clear();Gyroy.clear();Gyroz.clear();
            input_signal.clear();
            byteBuffer.clear();
        }
        model.close();
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {
    }

    public static float round(float d) {
        BigDecimal bd = new BigDecimal(Float.toString(d));
        bd = bd.setScale(2, BigDecimal.ROUND_HALF_UP);
        return bd.floatValue();
    }

    private static List<Float> normalize(List<Float> list)
    {
        float mean = mean(list); float var = var(list);
        int i;
        for(i=0; i < list.size(); i++)
        {
            list.set(i,((list.get(i) - mean)/(var+0.00001f)));
        }
        return  list;
    }


    public static float var(List<Float> list) {
        Float mean = mean(list);
        float var=0.0f;
        for (Float i: list) {
            var += Math.pow(i-mean,2);
        }
        var= (float) Math.sqrt(var);
        return var;
    }

    public static float mean(List<Float> list) {
        float sum = 0.0f;
        for (Float i: list) {
            sum += i;
        }
        return sum/((float)list.size());
    }

}