using System;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using static AnimeGANv2_Onnx_Sharp.OpenCVSharpWrap;
using AnimeGANv2_Onnx_Sharp;

namespace ComicFaceOnnx
{
   public class AnimeGAN
   {


      public static bool IsOnnxLoaded=false;

      private static InferenceSession _session;
      private static SessionOptions _option;
      private static List<NamedOnnxValue> _container = new List<NamedOnnxValue>();
      public static async Task<bool> InitLoadOnnx()
      {
         IsOnnxLoaded = false;
         await Task.Run(() =>
         {
            _option = new SessionOptions
            {
               LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
            };
            _option.AppendExecutionProvider_CPU(0);
            _session = new InferenceSession(onnx_resource.face_paint_512_v2_0, _option);
            IsOnnxLoaded = true;
         });
         return IsOnnxLoaded;

      }
      private static DenseTensor<float> PreInput(Mat mat)
      {

         var mat512 = new Mat();
         Cv2.Resize(mat, mat512, new Size(512, 512));

         Mat m512 = new Mat(512,512,MatType.CV_8UC3);

         Cv2.CvtColor(mat512, m512, ColorConversionCodes.BGR2RGB);

         //Mat m512 = mat512;

         //输入Tensor
         DenseTensor<float> input_tensor = new DenseTensor<float>(new[] { 1, 3, 512, 512 });
         // 输入Tensor
         for (int y = 0; y < m512.Height; y++)
         {
            for (int x = 0; x < m512.Width; x++)
            {

               input_tensor[0, 0, y, x] = (mat512.At<Vec3b>(y, x)[0] / 255f - 0.5f) / 0.5f;
               input_tensor[0, 1, y, x] = (mat512.At<Vec3b>(y, x)[1] / 255f - 0.5f) / 0.5f;
               input_tensor[0, 2, y, x] = (mat512.At<Vec3b>(y, x)[2] / 255f - 0.5f) / 0.5f;
            }
         }
         return input_tensor;
      }
      public static async Task<Mat> RunB2M(int w, int h, byte[] bytes)
      {


         Mat mat = Bytes2Mat8UC3(w, h, bytes);

         var inputTensor = PreInput(mat);//DenseTensor<float> inputTensor
         Mat result_mat = new Mat();
         await Task.Run(() =>
         {

            _container.Clear();
            _container.Add(NamedOnnxValue.CreateFromTensor<float>("input_image", inputTensor));
            var results = _session.Run(_container);
            var resultArrays = results.ToArray();

            var result_tensors = results[0].AsTensor<float>();

            var result_array = result_tensors.ToArray();

            float[] temp_r = new float[512 * 512];
            float[] temp_g = new float[512 * 512];
            float[] temp_b = new float[512 * 512];

            Array.Copy(result_array, temp_r, 512 * 512);
            Array.Copy(result_array, 512 * 512, temp_g, 0, 512 * 512);
            Array.Copy(result_array, 512 * 512 * 2, temp_b, 0, 512 * 512);

            Mat rmat = new Mat(512, 512, MatType.CV_32FC1, temp_r);
            Mat gmat = new Mat(512, 512, MatType.CV_32FC1, temp_g);
            Mat bmat = new Mat(512, 512, MatType.CV_32FC1, temp_b);

            rmat = (rmat + 1f) * 127.5f;
            gmat = (gmat + 1f) * 127.5f;
            bmat = (bmat + 1f) * 127.5f;


            Cv2.Merge(new Mat[] { rmat, gmat, bmat }, result_mat);
         });
         return result_mat;
      }
      public static async Task<byte[]> RunB2B(int w, int h, byte[] bytes)
      {



         Mat result_mat = new Mat();
         await Task.Run(() =>
         {
            Mat mat = Bytes2Mat8UC3(w, h, bytes);

            var inputTensor = PreInput(mat);


            _container.Clear();
            _container.Add(NamedOnnxValue.CreateFromTensor<float>("input_image", inputTensor));
            var results = _session.Run(_container);
            var resultArrays = results.ToArray();

            var result_tensors = results[0].AsTensor<float>();

            var result_array = result_tensors.ToArray();

            float[] temp_r = new float[512 * 512];
            float[] temp_g = new float[512 * 512];
            float[] temp_b = new float[512 * 512];

            Array.Copy(result_array, temp_r, 512 * 512);
            Array.Copy(result_array, 512 * 512, temp_g, 0, 512 * 512);
            Array.Copy(result_array, 512 * 512 * 2, temp_b, 0, 512 * 512);

            Mat rmat = new Mat(512, 512, MatType.CV_32FC1, temp_r);
            Mat gmat = new Mat(512, 512, MatType.CV_32FC1, temp_g);
            Mat bmat = new Mat(512, 512, MatType.CV_32FC1, temp_b);

            rmat = (rmat + 1f) * 127.5f;
            gmat = (gmat + 1f) * 127.5f;
            bmat = (bmat + 1f) * 127.5f;


            Cv2.Merge(new Mat[] { rmat, gmat, bmat }, result_mat);
         });
         return result_mat.ToBytes();
      }
      /// <summary>
      /// OpenCV mat input and output , use opencvsharp pkg in your own prj too
      /// </summary>
      /// <param name="mat"></param>
      /// <returns></returns>
      public static async Task<Mat> RunM2M(Mat mat)
      {
         var inputTensor = PreInput(mat);//DenseTensor<float> inputTensor
         Mat result_mat = new Mat();
         await Task.Run(() =>
         {

            _container.Clear();
            _container.Add(NamedOnnxValue.CreateFromTensor<float>("input_image", inputTensor));
            var results = _session.Run(_container);
            var resultArrays = results.ToArray();

            var result_tensors = results[0].AsTensor<float>();

            var result_array = result_tensors.ToArray();

            float[] temp_r = new float[512 * 512];
            float[] temp_g = new float[512 * 512];
            float[] temp_b = new float[512 * 512];

            Array.Copy(result_array, temp_r, 512 * 512);
            Array.Copy(result_array, 512 * 512, temp_g, 0, 512 * 512);
            Array.Copy(result_array, 512 * 512 * 2, temp_b, 0, 512 * 512);

            Mat rmat = new Mat(512, 512, MatType.CV_32FC1, temp_r);
            Mat gmat = new Mat(512, 512, MatType.CV_32FC1, temp_g);
            Mat bmat = new Mat(512, 512, MatType.CV_32FC1, temp_b);

            rmat = (rmat + 1f) * 127.5f;
            gmat = (gmat + 1f) * 127.5f;
            bmat = (bmat + 1f) * 127.5f;


            Cv2.Merge(new Mat[] { rmat, gmat, bmat }, result_mat);
         });
         return result_mat;
      }


      /// <summary>
      /// bytes input / bytes output , so no denpent
      /// </summary>
      /// <param name="w"></param>
      /// <param name="h"></param>
      /// <param name="bytes">bytes input [4 Channel]</param>
      /// <param name="resize_back"></param>
      /// <returns></returns>
      public static async Task<byte[]> RunB2B(int w, int h, byte[] bytes,bool resize_back=false)
      {

         Mat result_mat = new Mat();
         await Task.Run(() =>
         {
            Mat mat = Bytes2Mat8UC3(w, h, bytes);

            var inputTensor = PreInput(mat);


            _container.Clear();
            _container.Add(NamedOnnxValue.CreateFromTensor<float>("input_image", inputTensor));
            var results = _session.Run(_container);
            var resultArrays = results.ToArray();

            var result_tensors = results[0].AsTensor<float>();

            var result_array = result_tensors.ToArray();

            float[] temp_r = new float[512 * 512];
            float[] temp_g = new float[512 * 512];
            float[] temp_b = new float[512 * 512];

            Array.Copy(result_array, temp_r, 512 * 512);
            Array.Copy(result_array, 512 * 512, temp_g, 0, 512 * 512);
            Array.Copy(result_array, 512 * 512 * 2, temp_b, 0, 512 * 512);

            Mat rmat = new Mat(512, 512, MatType.CV_32FC1, temp_r);
            Mat gmat = new Mat(512, 512, MatType.CV_32FC1, temp_g);
            Mat bmat = new Mat(512, 512, MatType.CV_32FC1, temp_b);

            rmat = (rmat + 1f) * 127.5f;
            gmat = (gmat + 1f) * 127.5f;
            bmat = (bmat + 1f) * 127.5f;


            Cv2.Merge(new Mat[] { rmat, gmat, bmat }, result_mat);
         });

         //Cv2.CvtColor(result_mat,result_mat,ColorConversionCodes.RGB2BGRA);
         if (resize_back)
         {


            Cv2.Resize(result_mat, result_mat,new Size(512,Convert.ToInt32(512*h/w)));
            //result_mat.Resize(h,w);
            //result_mat.Resize(,, 512,0);
         }



         return result_mat.ToBytes();
      }
      /// <summary>
      /// hidden / won't work
      /// </summary>
      /// <param name="w"></param>
      /// <param name="h"></param>
      /// <param name="bytes"></param>
      /// <param name="resize_back"></param>
      /// <returns></returns>
      private static async Task<byte[]> RunB2BC3(int w, int h, byte[] bytes, bool resize_back = false)
      {

         Mat result_mat = new Mat();
         await Task.Run(() =>
         {
            //byte[] copy = new byte[bytes.Length];
            //Array.Copy(bytes, copy, bytes.Length);
            Mat mat = new Mat(h, w, MatType.CV_8UC3, bytes);
            mat = mat.CvtColor(ColorConversionCodes.BGR2RGB, 3);
            var inputTensor = PreInput(mat);


            _container.Clear();
            _container.Add(NamedOnnxValue.CreateFromTensor<float>("input_image", inputTensor));
            var results = _session.Run(_container);
            var resultArrays = results.ToArray();

            var result_tensors = results[0].AsTensor<float>();

            var result_array = result_tensors.ToArray();

            float[] temp_r = new float[512 * 512];
            float[] temp_g = new float[512 * 512];
            float[] temp_b = new float[512 * 512];

            Array.Copy(result_array, temp_r, 512 * 512);
            Array.Copy(result_array, 512 * 512, temp_g, 0, 512 * 512);
            Array.Copy(result_array, 512 * 512 * 2, temp_b, 0, 512 * 512);

            Mat rmat = new Mat(512, 512, MatType.CV_32FC1, temp_r);
            Mat gmat = new Mat(512, 512, MatType.CV_32FC1, temp_g);
            Mat bmat = new Mat(512, 512, MatType.CV_32FC1, temp_b);

            rmat = (rmat + 1f) * 127.5f;
            gmat = (gmat + 1f) * 127.5f;
            bmat = (bmat + 1f) * 127.5f;


            Cv2.Merge(new Mat[] { rmat, gmat, bmat }, result_mat);
         });


         if (resize_back)
         {


            Cv2.Resize(result_mat, result_mat, new Size(512, Convert.ToInt32(512 * h / w)));
            //result_mat.Resize(h,w);
            //result_mat.Resize(,, 512,0);
         }



         return result_mat.ToBytes();
      }


   }

}
