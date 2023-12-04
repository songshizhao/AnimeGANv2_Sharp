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
      private static InferenceSession _session;
      private static SessionOptions _option;
      private static List<NamedOnnxValue> container = new List<NamedOnnxValue>();
      public static async Task<bool> InitLoadOnnx()
      {
         bool success = false;
         await Task.Run(() =>
         {
            _option = new SessionOptions
            {
               LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
            };
            _option.AppendExecutionProvider_CPU(0);
            _session = new InferenceSession(onnx_resource.face_paint_512_v2_0, _option);
            success = true;
         });
         return success;

      }
      private static DenseTensor<float> PreInput(Mat mat)
      {

         var mat512 = new Mat();
         Cv2.Resize(mat, mat512, new Size(512, 512));

         Mat m512 = new Mat();

         Cv2.CvtColor(mat512, m512, ColorConversionCodes.BGR2RGB);

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

            container.Clear();
            container.Add(NamedOnnxValue.CreateFromTensor<float>("input_image", inputTensor));
            var results = _session.Run(container);
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


            container.Clear();
            container.Add(NamedOnnxValue.CreateFromTensor<float>("input_image", inputTensor));
            var results = _session.Run(container);
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
      public static async Task<Mat> RunM2M(Mat mat)
      {
         var inputTensor = PreInput(mat);//DenseTensor<float> inputTensor
         Mat result_mat = new Mat();
         await Task.Run(() =>
         {

            container.Clear();
            container.Add(NamedOnnxValue.CreateFromTensor<float>("input_image", inputTensor));
            var results = _session.Run(container);
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
   }

}
