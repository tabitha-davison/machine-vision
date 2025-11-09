import easyocr, cv2

def easy_ocr():

  easy_ocr_image =cv2.imread("IMG_3638.jpg")

  reader = easyocr.Reader(['en'], gpu=True)
  easy_ocr_result = reader.readtext(easy_ocr_image, detail=1, paragraph = False)
  print(easy_ocr_result)

  for (coord, text, prob) in easy_ocr_result:
    (top_left, top_right, bottom_right, bottom_left) = coord
    tx, ty = (int(top_left[0]), int(top_left[1]))
    bx, by = (int(bottom_right[0]), int(bottom_right[1]))
    cv2.rectangle(easy_ocr_image, (tx, ty), (bx, by), (0, 255, 0), 2)
    cv2.putText(easy_ocr_image, text, (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

  return cv2.imwrite("easy_ocr_result.png", easy_ocr_image)

# Talk about the pros and cons and how this algorithm actually works well