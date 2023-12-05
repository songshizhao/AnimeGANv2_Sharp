using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace AnimeGANv2_Onnx_Sharp
{
   public static class OpenCVSharpWrap
   {

      //

      public static Mat Bytes2Mat8UC3(int w, int h, byte[] bytes)
      {
         Mat mat = new Mat(h, w, MatType.CV_8UC4, bytes);

         Mat[] channels=mat.Split();
         Mat[] tomerge = new Mat[] { channels[0], channels[1],channels[2]};//BGR
         Mat result=new Mat();


         Cv2.Merge(tomerge, result);
         //mat = mat.CvtColor(ColorConversionCodes.BGRA2BGR, 3);
         //mat = mat.CvtColor(ColorConversionCodes.BGRA2RGB, 3);
        
         return result;
      }

      public static byte[] Mat2Bytes(Mat mat)
      {
         return mat.ToBytes();
      }

      public static async Task<Mat> Smooth(Mat sourceMat, double lambda = 2e-2, double kappa = 2.0)
      {

         Mat output = new Mat();//sourceMat.Width,sourceMat.Height,MatType.CV_8UC4

         sourceMat = sourceMat.CvtColor(ColorConversionCodes.BGRA2BGR, 3);
         await Task.Run(() => {


            OpenCvSharp.XImgProc.CvXImgProc.L0Smooth(sourceMat, output, lambda, kappa);

         });

         return output;

      }

      /// <summary>
      /// smooth need 3 channel input
      /// </summary>
      /// <param name="w"></param>
      /// <param name="h"></param>
      /// <param name="bytes"></param>
      /// <returns></returns>
      public static async Task<byte[]> L0SmoothC3(int w, int h, byte[] bytes)
      {
         Mat mat = new Mat(h, w, MatType.CV_8UC3, bytes);
         //mat = mat.CvtColor(ColorConversionCodes.BGRA2BGR, 3);
         Mat output=await Smooth(mat);

         return output.ToBytes();

      }
      ///// <summary>
      ///// 
      ///// </summary>
      ///// <param name="w"></param>
      ///// <param name="h"></param>
      ///// <param name="bytes"></param>
      ///// <returns></returns>
      //public static async Task<byte[]> L0SmoothC4(int w, int h, byte[] bytes)
      //{
      //   Mat mat = new Mat(h, w, MatType.CV_8UC4, bytes);
      //   mat = mat.CvtColor(ColorConversionCodes.BGRA2BGR, 3);
      //   Mat output = await Smooth(mat);

      //   return output.ToBytes();

      //}







      /// <summary>
      /// 4 channel -> 3 channel then smooth
      /// </summary>
      /// <param name="w"></param>
      /// <param name="h"></param>
      /// <param name="bytes"></param>
      /// <param name="needC4back">need convert c4 back</param>
      /// <returns></returns>
      public static async Task<byte[]> L0SmoothC4(int w, int h, byte[] bytes,bool needC4back=false)
      {

         Mat mat = new Mat(h, w, MatType.CV_8UC4, bytes);
         mat = mat.CvtColor(ColorConversionCodes.BGRA2BGR, 3);
         Mat output = await Smooth(mat);


         if (needC4back)
         {
            mat = mat.CvtColor(ColorConversionCodes.BGR2BGRA, 4);
         }
         return output.ToBytes();

      }

   }
}
