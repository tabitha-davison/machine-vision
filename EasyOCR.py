import easyocr, cv2


def easy_ocr(easy_ocr_image):
    full_text = []
    reader = easyocr.Reader(["en"], gpu=True)
    easy_ocr_result = reader.readtext(easy_ocr_image, detail=1, paragraph=False)

    for coord, text, prob in easy_ocr_result:
        full_text.append(text)
        (top_left, top_right, bottom_right, bottom_left) = coord
        tx, ty = (int(top_left[0]), int(top_left[1]))
        bx, by = (int(bottom_right[0]), int(bottom_right[1]))
        cv2.rectangle(easy_ocr_image, (tx, ty), (bx, by), (0, 255, 0), 2)
        cv2.putText(
            easy_ocr_image,
            text,
            (tx, ty - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    return full_text
