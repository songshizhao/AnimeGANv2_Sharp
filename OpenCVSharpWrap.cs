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
         mat = mat.CvtColor(ColorConversionCodes.BGRA2BGR, 3);
         return mat;
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








   }
}
