=== LCD OCR ===

Finds LCD display on a digital photo and prepares it's fragments for OCR.  
Test photos are made on a low end 3MP smartphone camera with no hardware autofocus.  
LCD display assumed to be the largest rectangle on the photo.  
Character segmentation are made based on vertical or horizontal projection of the image. To help analyze the projection, 
 the ratio of second derivative of this curve to its height is calculated (PAMI96, p.9)



=== Chain Of Responsibility design pattern ===  
visible: Video > Photo > LCD > SingleRow  
croppable: CroppableImage > AreaFactory > Projection  
