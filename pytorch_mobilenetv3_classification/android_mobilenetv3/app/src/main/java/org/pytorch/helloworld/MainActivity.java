package org.pytorch.helloworld;
//how to load a pre-trained PyTorch model and use it for inference to classify images
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.StrictMode;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.MemoryFormat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

  Module module = null;
  Bitmap bitmap = null;
  int bitmap_index = 1;
  private ImageView imageView;
  private TextView textView;
  private ArrayList<String> classNames;


//  线程策略来关闭或调整 Android 的严格模式。在这里，它配置为检测所有线程策略的违规行为
  public static void closeStrictMode() {
    StrictMode.setThreadPolicy(new StrictMode.ThreadPolicy.Builder()
            .detectAll().penaltyLog().build());
  }
  @Override
//  表示这个方法覆盖了父类的方法。 设置活动的内容视图为 activity_main.xml
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    try {

      module = LiteModuleLoader.load(assetFilePath(this, "mobile.ptl"));
    } catch (IOException e) {
      e.printStackTrace();
    }
    // 获取控件
    imageView = findViewById(R.id.image);
    textView = findViewById(R.id.text);
    Button selectImgBtn = findViewById(R.id.select_img_btn);
    Button openCamera = findViewById(R.id.open_camera);
//设置 "Select Image" 按钮的点击监听器加载一个图片文件，并将其显示在 imageView 上。图片的索引通过 bitmap_index 控制，每次点击后图片索引增加，达到 11 时重置为 0
    selectImgBtn.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        //
        try {
          bitmap = BitmapFactory.decodeStream(getAssets().open(bitmap_index+".jpg"));
        } catch (IOException e) {
          e.printStackTrace();
        }
        if(bitmap_index==11)
          bitmap_index=0;
        bitmap_index++;
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
        imageView.setImageBitmap(resizedBitmap);

        // preparing input tensor 行代码
        // 将加载的图片转换成 PyTorch Tensor，并将其作为输入传递给模型。bitmapToFloat32Tensor 方法用于将 Bitmap 对象转换为模型可接受的输入格式，并进行了标准化处理。

        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);

        // running the model
        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        // getting tensor content as java array of floats
        final float[] scores = outputTensor.getDataAsFloatArray();

        // searching for the index with maximum score
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = 0;
        for (int i = 0; i < scores.length; i++) {
          if (scores[i] > maxScore) {
            maxScore = scores[i];
            maxScoreIdx = i;
          }
        }

        String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];

        // showing className on UI
        TextView textView = findViewById(R.id.text);
        textView.setText(className);
      }
    });
    openCamera.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        // 打开实时拍摄识别页面
        Intent intent = new Intent(MainActivity.this, CameraActivity.class);
        startActivity(intent);
      }
    });

  }

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */

//  This utility method copies an asset to the internal storage of the app and returns the absolute path of the copied file.
//  It is used to access the PyTorch model file.
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }
}
