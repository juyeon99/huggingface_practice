# https://github.com/JaidedAI/EasyOCR

# STEP 1: import modules
import easyocr

# STEP 2: create inference object
reader = easyocr.Reader(['ko','en'])

# STEP 3: load data
data = 'newyr.png'

# STEP 4: inference
result = reader.readtext(data, detail=0)
print(result)

# STEP 5: post-processing
# if sin == "주민등록등본":
#     ~~~~