import numpy as np
import cv2

#Funkcja skalująca obraz wejściowy do rozmiarów pozwalających sie wyświetlić w znośniej formie
def resize(img,scale):
	width = int(img.shape[1] * scale)
	height = int(img.shape[0] * scale)
	dimensions = (width, height)
	resized_img = cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)
	return resized_img

#Wstępny proces przekształcenia obrazu w celu możliwości wyłuskania konturu karty niezależnie czy użyto
#pieprzu czy gradientu, jednocześnie uwzględnia bliskie ułożenie kart, więc jest dobrany tak, aby się  nie zlewały kontury
def convert(resized_img):
	gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
	median = cv2.medianBlur(gray, 3)  # Filtrowanie medianowe z jądrem o rozmiarze 5x5
	kernel = np.ones((5, 5), np.uint8)
	erode = cv2.erode(median, kernel, iterations=1)
	thresh = cv2.adaptiveThreshold(erode, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
	canny = cv2.Canny(thresh, 100, 200)
	cv2.imshow('Convert', thresh)
	#cv2.imshow('Skalowane', thresh)
	return canny,median

#Funkcja wyszukująca kontury na głównym obrazie, które nie znajdują się w żadnym innym konturze, czyli wynajduje kontury kart
def find_contours(canny):
	kontury, hierarchia = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# Filtrowanie konturów, które nie są wewnątrz innych konturów
	filtered_contours = []
	for i, kontur in enumerate(kontury):
		if hierarchia[0][i][3] == -1:  # Jeśli indeks konturu nadrzędnego w hierarchii jest równy -1, oznacza to, że kontur nie jest wewnątrz innego konturu
			filtered_contours.append(kontur)
	return filtered_contours

# Zawarta w funkcji 'rotate'. Z racji na różnie obrócone karty na zdjęciu wejściowym, należy przesegregować współrzędne rogów
# tak, aby karta zawsze była obrócona pionowo
def sort(box):
	sorted_box = sorted(box, key=lambda coord: coord[0] + coord[1])
	x2, y2 = sorted_box[1]
	x3, y3 = sorted_box[2]
	if y2 > y3:
		tmp = sorted_box[1]
		sorted_box[1] = sorted_box[2]
		sorted_box[2] = tmp
		print(sorted_box[0])
		print(sorted_box)
	return sorted_box

# Zawarta w funkcji 'rotate'. Wycina obszar, który zawiera kontur, dzięki któremu zostanie zidentfikowana karta
def mask(width,height,masked_card):
	center = (width // 2, height // 2)
	mask = np.ones(masked_card.shape, np.uint8)
	cv2.ellipse(mask, center, (100, 70), -60, 0, 360, (255, 255, 255), -1)
	mask = cv2.bitwise_not(mask)
	masked_card = cv2.add(masked_card, mask)
	return masked_card

# Funkcja, której zadaniem jest obrócenie pojedynczej karty, przeskalowaniu jej do rozmiarów określonych
# oraz podaniu na wyjściu już przeskalowanego obrócnego obrazu
def rotate(filtered_contours, resized_img):
	for i, kontur in enumerate(filtered_contours):
		if (cv2.arcLength(kontur, True) > 500):
			accuracy = 0.05 * cv2.arcLength(kontur, True)
			approx = cv2.approxPolyDP(kontur, accuracy, True)
			#obrazek = cv2.drawContours(resized_img, approx, 1, (255, 0, 0), 5)
			rect = cv2.minAreaRect(approx)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			cv2.polylines(resized_img, [box], True, (255, 0, 0), 2)
			sorted_box=sort(box)
			#cv2.imshow('kontury', obrazek)
			width, height = 210, 300
			box_1 = np.float32(sorted_box)
			box_2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
			matrix = cv2.getPerspectiveTransform(box_1, box_2)
			rotated_img = cv2.warpPerspective(resized_img, matrix, (width, height))
			masked_card = rotated_img.copy()
			masked_card=mask(width,height,masked_card)
			cv2.imshow(f'Maska {i}', masked_card)
			recognize_char(masked_card,i)
			#cv2.imshow(f'Obrocone karty {i}', rotated_img)

			# if i == 0:
			# 	card_1 = rotated_img
			# 	mask_1 = masked_card1
			# 	print(1)
			# elif i == 1:
			# 	card_2 = rotated_img
			# 	mask_2 = masked_card1
			# 	print(2)
			# elif i == 2:
			# 	card_3 = rotated_img
			# 	mask_3 = masked_card1
			# 	print(3)
			# elif i == 3:
			# 	card_4 = rotated_img
			# 	mask_4 = masked_card1
			# 	print(4)

# Komentarze wewnątrz funkcji
# W pierwszej kolejności dla pewnego obszaru wyznaczany jest kolor karty dla poszczególnych zakresów
# Następnie wewnętrzny kontur z obrazka, znaku odróżniającego karty jest analizowany pod względem 3 parametrów
# długości, powierzchni i stosunkowi wysokości do szerokości konturu
# Kolejno nakreślono zakresy dla których dany kontur określa daną kartę
# Infromacja o karcie wypisywana jest na niej
def recognize_char(masked_card,i):
	canny,ero=prepare_contour(masked_card)
	cv2.imshow(f'Kontury do rozpoznania {i}', canny)
	color = "nic"
	hsv = cv2.cvtColor(masked_card, cv2.COLOR_BGR2HSV)
	h_values = hsv[100:170, 70:90, 0]
	s_values = hsv[100:170, 70:90, 1]
	v_values = hsv[100:170, 70:90, 2]
	h = np.mean(h_values)
	s = np.mean(s_values)
	v = np.mean(v_values)

	if (h > 52 and h < 81 and s > 63 and s < 91 and v > 135 and v < 172):
		color = "Kolor zloty"
	if (h > 75 and h < 125 and s > 85 and s < 156 and v > 85 and v < 170):
		color = "Kolor czerwony"
	if (h > 80 and h < 110 and ((s > 76 and s < 152)or (s>170 and s<196)) and ((v > 115 and v < 155)or(v > 180 and v < 195))):
		color = "Kolor niebieski"
	if (h > 78 and h < 91 and s > 89 and s < 146 and ((v > 77 and v < 101)or(v > 148 and v < 155))):
		color = "Kolor zielony"


	kontury, hierarchia = cv2.findContours(ero, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# obrysowuje kontur znaczka
	if len(kontury) >= 2:
		cnt = kontury[2]

		if cv2.arcLength(cnt, True) > 100:
			area = cv2.contourArea(cnt)
			perimeter = cv2.arcLength(cnt, True)
			x, y, w, h = cv2.boundingRect(cnt)
			aspect_ratio = float(h) / w
			img_cnt=masked_card
			#img_cnt = cv2.drawContours(masked_card, [cnt], -1, (0, 255, 255), 5)
			#cv2.imshow(f'Identyfikowane kontury {i}', img_cnt)
			if area>4500:
				text = "Pauza"
				cv2.putText(img_cnt, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 100), 1, cv2.LINE_AA)
			elif (area<1000 and perimeter<200) or (area>2490 and area<2730 and perimeter>260 and perimeter<285):
				text = "Zmiana kierunku "
				cv2.putText(img_cnt, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 100), 1, cv2.LINE_AA)
			elif (perimeter>295 and perimeter <315) or (perimeter>254 and perimeter<281 and area > 920 and area <1472):
				text = "Karta numer 7 "
				cv2.putText(img_cnt, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 100), 1, cv2.LINE_AA)
			elif (perimeter>430 and perimeter<455 and area>2700 and area < 4000) or (perimeter>421 and perimeter<427 and aspect_ratio>1.36):
				text = "Karta numer 5 "
				cv2.putText(img_cnt, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 100), 1, cv2.LINE_AA)
			else:
				text = "Karta numer 3 "
				cv2.putText(img_cnt, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 100), 1, cv2.LINE_AA)
			cv2.putText(img_cnt, color, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 100), 1, cv2.LINE_AA)

			print(f"Powierzchnia konturu: {area}")
			print(f"Długość konturu: {perimeter}")
			print(f"Stosunek wysokości do długości prostokąta: {aspect_ratio}")
			cv2.imshow(f'Rozpoznana karta {i}', img_cnt)

	return ero

#Funkcja przygotowuje ROI zakreślone przez funkcję mask do łatwej detekcji i wskazania konturów
def prepare_contour(card):
    card=cv2.cvtColor(card, cv2.COLOR_BGR2GRAY, card)
    card=cv2.medianBlur(card, 11)
    gaussian_card=cv2.GaussianBlur(card, (17, 17), 30)
    card=gaussian_card
    gaussian_card=cv2.bitwise_not(gaussian_card)
    cv2.equalizeHist(gaussian_card, card)
    card=cv2.bitwise_not(card)
    cv2.threshold(card, 0, 255, cv2.THRESH_OTSU, card)
    erode = cv2.dilate(card, (5,5), iterations=1)
    erode = cv2.erode(erode, (13,13), iterations=1)
    canny = cv2.Canny(erode, 120, 255, 5)
    return canny, erode



img = cv2.imread('C:/Users/frane/OneDrive/Desktop/6.png')
scale = 0.5
resized_img = resize(img,scale)
canny,median = convert(resized_img)
filtered_contours=find_contours(canny)
rotate(filtered_contours,resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()






