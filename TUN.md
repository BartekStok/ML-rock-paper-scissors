### For R.N. only:

W procesie Machine Learning do rozpoznawania zdjęć rąk 
wykorzystałem biblioteki:
 
 $ `Pillow` do odczytu i obróki zdjęć, np: `PIL.image.open(path)`,
 `PIL.Image.resize(image)` 
 
$ `joblib` do zapisu i odczytu obrobionych zdjęć w jednym 
pilku w formacie pkl.

$ `Pandas` do utworzenia pożadanego formatu danych (DataFrame) 
z odczytanego pliku. 

$ `NumPy` do utworzenia macierzy z DataFrame. Różne zabiegi
na zestawie danych były konieczne, by doprowadzić do kształtu
macierzy mogącego posłużyć do procesu ML. Początkowo dane są
w kształcie `[nr_zdjęcia, zdjęcie(PX, PY, color), klasa]`. To jest macierz, gdzie
nr_zdjęcia to indeks, zdjęcie to lista PX x PY x color, klasa to 
klasyfikator.

Tą macierz podzieliłem na dwie wg ogólnej konwencji:

X = dane do analizy

y = klasa

Macierz X potrzebowała jeszcze obróbki, początkowo była w postaci
[ilość_zdjęć, PX, PY, color], a ostatecznie doprowadziłem ją do
postaci [ilość_zdjęć, macierz_zdjęcia]. Macierz zdjęcia uzyskałem
łącząc PX, PY i color, a następnie wypłaszczając macierz.
Można tu używać gotowych bibliotek w SciKit Learn. 

Gdy już miałem gotowe X i y mogłem przystąpić do wybrania
włąsciwego klasyfikatora.

$ `SciKit Learn` do cross walidacji, trenowania modelu.
Cross-Validation zrobiłem dzieląc zestaw danych na 5 równych
części, i następnie w pięciu krokach dzieliłem na część do 
trenowania i testowania modelu, tak by każda część była i w treningu
i w testowaniu. Następnie dla każdego kroku trenowałem model
i zapisywałem jego wskaźniki Accuracy, Precision, Recall(Sensitivity),
i F1 score. Z tych danych powstał wykres, na którym można było
określić który klasyfikator ma najlepszą precyzję (bo o precyzję nam chodzi).

Wybrałem model SGD (Stochastic Gradient Descent).

Następnie już dla całego zestawu danych przeprowadziłem trening
finalnego modelu. 

$ `matplotlib` do rysowania wykresów, pokazywania przykładowych
zdjęć.

