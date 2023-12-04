# AnimeGANv2_Sharp

```csharp

      private async void MainPage_Loaded(object sender, RoutedEventArgs e)
      {

         //must load onnx before run 
         bool r =await AnimeGAN.InitLoadOnnx();
         Debug.WriteLine("load onnx result: "+r.ToString());
      }
      private async void select_Click(object sender, RoutedEventArgs e)
      {
         FileOpenPicker fileOpenPicker = new FileOpenPicker();
         fileOpenPicker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
         fileOpenPicker.FileTypeFilter.Add(".jpg");
         fileOpenPicker.FileTypeFilter.Add(".png");
         fileOpenPicker.FileTypeFilter.Add(".jpeg");
         fileOpenPicker.FileTypeFilter.Add(".bmp");
         fileOpenPicker.SettingsIdentifier = "input single";
         fileOpenPicker.ViewMode = PickerViewMode.Thumbnail;
         var file = await fileOpenPicker.PickSingleFileAsync();
         if (file!=null)
         {
            BitmapDecoder bitmapDecoder = await BitmapDecoder.CreateAsync(await file.OpenReadAsync());
            //
            var pixels = await bitmapDecoder.GetPixelDataAsync();
            byte[] imageBytes = (byte[])pixels.DetachPixelData().Clone();
            int w = (int)bitmapDecoder.PixelWidth, h = (int)bitmapDecoder.PixelHeight;
            var outputBytes = await AnimeGAN.RunB2B(w, h, imageBytes);
            var bi = new BitmapImage();
            InMemoryRandomAccessStream ms = new InMemoryRandomAccessStream();
            await ms.WriteAsync(outputBytes.AsBuffer());
            ms.Seek(0);
            await bi.SetSourceAsync(ms);
            img3.Source = bi;


            Mat sourceMat = new Mat(h,w,MatType.CV_8UC4, imageBytes);
            ShowImage(img1,sourceMat);
            //smooth
            sourceMat=sourceMat.CvtColor(ColorConversionCodes.BGRA2BGR,3);
            var resultMat=await OpenCVSharpWrap.Smooth(sourceMat);
            ShowImage(img2,resultMat);

         }



      }
      public async void ShowImage(Windows.UI.Xaml.Controls.Image image, Mat mat)
      {
         var bi = new BitmapImage();
         await this.Dispatcher.RunAsync(Windows.UI.Core.CoreDispatcherPriority.Normal, async () => {
            byte[] bytes = mat.ToBytes();
            using (var ms = new InMemoryRandomAccessStream())
            {
               await ms.WriteAsync(bytes.AsBuffer());
               ms.Seek(0);
               await bi.SetSourceAsync(ms);
            }
         });
         image.Source = bi;
      }

 }
```


   
