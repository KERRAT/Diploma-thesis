import os
import cv2
import numpy as np
import tensorflow as tf
from imutils import contours
from tensorflow.keras.models import load_model
from Page_to_lines import get_lines, display_lines
import tempfile
def resize_image(image, max_size=960):
    height, width = image.shape[:2]

    if height > max_size or width > max_size:
        if height > width:
            new_height = max_size
            new_width = int((width * max_size) / height)
        else:
            new_width = max_size
            new_height = int((height * max_size) / width)
        return cv2.resize(image, (new_width, new_height))
    else:
        return image

def clear_border(image):
    top, bottom, left, right = 1, 1, 1, 1
    image_without_borders = image[top:-bottom, left:-right]

    image_with_border = cv2.copyMakeBorder(image_without_borders, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return image_with_border

def format_image(image):
    if image is None or not isinstance(image, np.ndarray) or len(image.shape) < 2:
        raise ValueError("Invalid input image.")

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] > 3:
        image = image[:, :, :3]
    return image

def clear_background(image):
    formated = format_image(image)

    image_with_border = clear_border(formated)

    gray_image = cv2.cvtColor(image_with_border, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 27, 50)

    return thresh

def is_line_empty(line, threshold=0.03, gray_threshold=130):
    """
    Determine if a line contains text based on the number of non-white pixels.

    Args:
    line (numpy.ndarray): Image of the line.
    threshold (float): Threshold for the proportion of non-white pixels to consider a line as empty. Default is 0.01 (1%).
    gray_threshold (int): Gray level threshold to consider a pixel as non-white. Default is 200.

    Returns:
    bool: True if the line is empty, False otherwise.
    """

    non_white_pixels = np.count_nonzero(line < gray_threshold)
    total_pixels = line.size

    if non_white_pixels / total_pixels < threshold:
        print('true:', non_white_pixels / total_pixels)
        return True
    else:
        print('false:', non_white_pixels / total_pixels)
        return False



def segment_words(image, p_image, file_name, line_number):
    converted = cv2.bitwise_not(p_image)

    blurred_image = cv2.GaussianBlur(converted, (5, 5), 0)

    # Apply morphological dilation to connect words
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 12))
    dilated = cv2.dilate(blurred_image, kernel, iterations=1)

    cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]


    words_images = []
    if len(cnts) > 0:
        cnts, _ = contours.sort_contours(cnts, method="left-to-right")

        for c in cnts:
            area = cv2.contourArea(c)
            if area > 10:
                x, y, w, h = cv2.boundingRect(c)
                ROI = image[y:y+h, x:x+w]
                words_images.append(ROI)

    return words_images


def extract_letters(word_image, img_size):
    no_border = clear_border(word_image)
    _, otsu_threshold = cv2.threshold(no_border, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    dilated = cv2.dilate(otsu_threshold, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(dilated, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = word_image.copy()
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
    letters = []

    cnts_sorted, _ = contours.sort_contours(cnts, method="left-to-right")

    # Process each contour, resize or pad the images
    for cont in cnts_sorted:
        x, y, w, h = cv2.boundingRect(cont)
        if h > 0 and w > 0:
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

            letter = word_image[y:y+h, x:x+w]

            thresh = cv2.adaptiveThreshold(letter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 181, 40)

            if h > img_size or w > img_size:
                # Resize the letter image while keeping the aspect ratio
                aspect_ratio = min(float(w), float(h)) / max(float(w), float(h))

                if h > img_size:
                    h = img_size
                    w = int(aspect_ratio * h)
                else: # w > img_size
                    w = img_size
                    h = int(aspect_ratio * w)

                letter_processed = cv2.resize(thresh, (w, h), interpolation=cv2.INTER_AREA)

            # Calculate padding for the current letter
            pad_top = img_size - h
            pad_bottom = 2
            pad_left = (img_size - w) // 2
            pad_right = img_size - w - pad_left


            # Pad the letter image to match the maximum dimensions
            letter_processed = cv2.copyMakeBorder(thresh, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            letters.append(letter_processed)

    return letters

def recognize_letter(letter_image, model, attempts=3, min_probability=0.2):
    letters = [
        'А','Б','В','Г','Ґ','Д','Е','Є','Ж','З','И','І','Ї','Й','К',
        'Л','М','Н','О','П','Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ',
        'Ь','Ю','Я','а','б','в','г','ґ','д','е','є','ж','з','и','і',
        'ї','й','к','л','м','н','о','п','р','с','т','у','ф','х','ц',
        'ч','ш','щ','ь','ю','я','1','2','3','4','5','6','7','8','9',
        '0','№','%','@',',','.','?',':',';','"','!','(',')','-','\''
    ]

    for attempt in range(attempts):
        # Змінюємо розмір зображення літери до 32x32
        resized_letter = cv2.resize(letter_image, (32, 32), interpolation=cv2.INTER_AREA)

        blur_size = (3, 3)  # розмір ядра для гаусового блюру, можна змінювати за потреби
        blur_sigma = 0  # відхилення, якщо дорівнює 0, відхилення обчислюється автоматично
        letter_blured = cv2.GaussianBlur(resized_letter, blur_size, blur_sigma)

        # Перевіряємо кількість каналів у зображенні
        if len(letter_blured.shape) == 3:
            # Конвертуємо зображення у відтінки сірого, якщо воно кольорове
            gray_letter = cv2.cvtColor(letter_blured, cv2.COLOR_BGR2GRAY)
        else:
            gray_letter = letter_blured

        # Конвертуємо в float32 та нормалізуємо
        data = np.array(gray_letter, dtype=np.float32)
        data = np.expand_dims(data, axis=-1)
        data /= 255.0

        # Передбачаємо літеру за допомогою навченої моделі
        prediction = model.predict(np.array([data]))[0]
        predicted_index = np.argmax(prediction)
        probability = prediction[predicted_index]  # Отримуємо ймовірність передбаченого індексу
        predicted_letter = letters[predicted_index]  # Отримуємо передбачену літеру з масиву літер

        if probability >= min_probability:
            return predicted_letter, probability, predicted_index
        else:
            # Змінюємо зображення перед наступною спробою
            if attempt == 0:
                # Зменшення зображення
                scale_percent = 80
                width = int(letter_image.shape[1] * scale_percent / 100)
                height = int(letter_image.shape[0] * scale_percent / 100)
                dim = (width, height)
                letter_image = cv2.resize(letter_image, dim, interpolation=cv2.INTER_AREA)
            elif attempt == 1:
                # Збільшення зображення
                scale_percent = 120
                width = int(letter_image.shape[1] * scale_percent / 100)
                height = int(letter_image.shape[0] * scale_percent / 100)
                dim = (width, height)
                letter_image = cv2.resize(letter_image, dim, interpolation=cv2.INTER_AREA)
            elif attempt == 2:
                # Поворот на 180 градусів
                letter_image = cv2.rotate(letter_image, cv2.ROTATE_180)
        # Якщо тричі ймовірність залишається низькою, повертаємо "_"
    return "_", 0, -1

def process_letter(letter_image, letter_idx, model):
    predicted_letter, probability, predicted_index = recognize_letter(letter_image, model)
    return predicted_letter

def process_word(word_image, file_name, line_idx, word_idx, model, img_size):
    result = ''
    letters = extract_letters(word_image, img_size)
    for letter_idx, letter_image in enumerate(letters):
        result += process_letter(letter_image, file_name, line_idx, word_idx, letter_idx, model)
    return result + ' '

def process_line(line, file_name, line_idx, model, img_size):
    result = ''
    if not line.size == 0:
        p_line = clear_background(line)
        if not is_line_empty(p_line):
            words_images = segment_words(line, p_line, file_name, line_idx)
            for word_idx, word_image in enumerate(words_images):
                result += process_word(word_image, file_name, line_idx, word_idx, model, img_size)
    return result + ' '

def process_image(image_path, model, img_size):
    result = ''

    image = cv2.imread(image_path)

    # Resize the image before processing
    resized_image = resize_image(image)

    # Save the resized image in a temporary file
    tmp_file_descriptor, tmp_file_name = tempfile.mkstemp(suffix='.jpg')
    os.close(tmp_file_descriptor)
    cv2.imwrite(tmp_file_name, resized_image)

    lines = get_lines(tmp_file_name, kernel_size=17, sigma=3, theta=9, smooth_window_len=3, threshold=0.3, peak_min_distance=2)

    os.remove(tmp_file_name)

    # The temporary file will be deleted when the context manager exits

    for line_idx, line in enumerate(lines):
        result += process_line(line, image_path, line_idx, model, img_size)

    return result.lower()