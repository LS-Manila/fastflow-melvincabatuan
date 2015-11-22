package ph.edu.dlsu.fastflow;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.TextureView;
import android.widget.ImageView;

import java.io.IOException;
import java.util.List;

/**
 * Created by cobalt on 11/17/15.
 */
public class CameraActivity extends Activity implements TextureView.SurfaceTextureListener, Camera.PreviewCallback {

    static {
        System.loadLibrary("native_opencv_module");
    }

    private Camera mCamera;
    private TextureView tv;
    private byte[] videoSource;
    private ImageView imViewA;
    private Bitmap imageA;
    final boolean LOG_FRAME_RATE = true;
    private boolean bProcessing = false;
    private Handler mHandler=new Handler(Looper.getMainLooper());
    private double current_fps;
    private double total_fps; // for average frame rate computation


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.camera_activity_layout);


        tv = (TextureView) findViewById(R.id.preview);
        imViewA = (ImageView) findViewById(R.id.imageViewA);
        tv.setSurfaceTextureListener(this);
    }




    int i = 0;
    long now, oldnow, count = 0;



    @Override
    public void onPreviewFrame(byte[] data, Camera camera) {

        if(LOG_FRAME_RATE) {
            /// Measure frame rate:
            i++;
            now = System.nanoTime() / 1000;
            if (i > 3) {
                current_fps = (float)1000000L / (now - oldnow);
                Log.d("onPreviewFrame: ", "Measured: " + current_fps + " fps.");
                total_fps += current_fps;
                count++;

                if(count%10 == 0){ // Log average every 10 frames
                    Log.d("onPreviewFrame: ", "AVERAGE: " + total_fps/count + " fps after " + count +" frames." );
                }
            }



            oldnow = now;
        }


        if (mCamera != null){
            if(!bProcessing) {
                videoSource = data;
                mHandler.post(DoImageProcessing);
            }
        }
    }

    public native void process(Bitmap pTarget, byte[] pSource);

    private Runnable DoImageProcessing = new Runnable() {
        public void run() {
            bProcessing = true;
            process(imageA, videoSource);
            imViewA.invalidate();
            mCamera.addCallbackBuffer(videoSource);
            bProcessing = false;
        }
    };

    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {

        /// Use front-facing camera (if available)
        Camera.CameraInfo cameraInfo = new Camera.CameraInfo();

        for (int camNo = 0; camNo < Camera.getNumberOfCameras(); camNo++) {
            Camera.CameraInfo camInfo = new Camera.CameraInfo();
            Camera.getCameraInfo(camNo, camInfo);

            if (camInfo.facing==(Camera.CameraInfo.CAMERA_FACING_FRONT)) {
                mCamera = Camera.open(camNo);
            }
        }
        if (mCamera == null) { /// Xperia LT15i has no front-facing camera, defaults to back camera
            mCamera = Camera.open();
        }


        try{


            mCamera.setPreviewTexture(surface);
            mCamera.setPreviewCallbackWithBuffer(this);
            mCamera.setDisplayOrientation(0);

            Camera.Size size = findBestResolution(width,height);
            PixelFormat pixelFormat = new PixelFormat();
            PixelFormat.getPixelFormatInfo(mCamera.getParameters().getPreviewFormat(), pixelFormat);
            int sourceSize = size.width * size.height * pixelFormat.bitsPerPixel / 8;

            /// Camera size and video format
            Camera.Parameters parameters = mCamera.getParameters();
            parameters.setPreviewSize(size.width, size.height);
            parameters.setPreviewFormat(PixelFormat.YCbCr_420_SP);
            mCamera.setParameters(parameters);

            /// Video buffer and bitmaps
            videoSource = new byte[sourceSize];
            imageA = Bitmap.createBitmap(size.width, size.height, Bitmap.Config.ARGB_8888);
            imViewA.setImageBitmap(imageA);

            /// Queue video frame buffer and start camera preview
            mCamera.addCallbackBuffer(videoSource);
            mCamera.startPreview();

        } catch (IOException e){
            mCamera.release();
            mCamera = null;
            throw new IllegalStateException();
        }
    }



    private Camera.Size findBestResolution(int pWidth, int pHeight){
        List<Camera.Size> sizes = mCamera.getParameters().getSupportedPreviewSizes();
        Camera.Size selectedSize = mCamera.new Size(0,0);

        for(Camera.Size size: sizes){
            if ((size.width <= pWidth) && (size.height <= pHeight) && (size.width >= selectedSize.width) && (size.height >= selectedSize.height )){
                selectedSize = size;
            }
        }

        if((selectedSize.width == 0) || (selectedSize.height == 0)){
            selectedSize = sizes.get(0);
        }

        return selectedSize;
    }


    @Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {

    }

    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {

        // Release camera

        if (mCamera != null){
            mCamera.stopPreview();
            mCamera.release();

            mCamera = null;
            videoSource = null;

            imageA.recycle();; imageA = null;
        }
        return true;
    }

    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture surface) {

    }
}