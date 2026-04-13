#!/usr/bin/env python3
"""
LAB4: Optical Flow
Szablon startowy (wersja ćwiczeniowa)

Cele:
1. Wczytać plik wideo z linii komend.
2. Wyznaczyć dobre punkty do śledzenia.
3. Śledzić je metodą Lucas-Kanade.
4. Wizualizować trajektorie ruchu punktów.

Instrukcja:
- Uzupełnij wszystkie sekcje TODO.
- Najpierw uruchom detekcję punktów.
- Następnie zaimplementuj śledzenie.
- Na końcu dodaj wizualizację trajektorii.
"""

import argparse
import sys

import cv2
import numpy as np


# ============================================================
# TODO 1: Wczytanie pliku wideo
# ============================================================

def wczytaj_wideo(sciezka_wideo: str):
    """
    Otwiera plik wideo.

    TODO:
    - Użyj cv2.VideoCapture(...)
    - Sprawdź, czy plik został otwarty poprawnie
    - Zwróć obiekt przechwytywania wideo
    """
    cap = cv2.VideoCapture(sciezka_wideo)

    if not (cap.isOpened()):
        print(f"BŁĄD: Nie można otworzyć pliku wideo: {sciezka_wideo}", file=sys.stderr)
        sys.exit(1)

    return cap


# ============================================================
# TODO 2: Wyznaczanie dobrych punktów do śledzenia
# ============================================================

def wykryj_punkty(obraz_szary: np.ndarray):
    """
    Wykrywa punkty charakterystyczne do śledzenia.

    TODO:
    - Użyj cv2.goodFeaturesToTrack(...)
    - Dobierz parametry, np.:
        * maxCorners
        * qualityLevel
        * minDistance
        * blockSize
    - Zwróć wykryte punkty
    """
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    punkty = cv2.goodFeaturesToTrack(
        obraz_szary,
        **feature_params
    )

    return punkty


# ============================================================
# TODO 3: Śledzenie punktów metodą Lucas-Kanade
# ============================================================

def sledz_punkty(poprzedni_obraz, biezacy_obraz, poprzednie_punkty):
    """
    Śledzi punkty pomiędzy dwiema kolejnymi klatkami.

    TODO:
    - Użyj cv2.calcOpticalFlowPyrLK(...)
    - Zwróć:
        * nowe_punkty
        * status
        * blad
    """
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    nowe_punkty, status, blad = cv2.calcOpticalFlowPyrLK(
        poprzedni_obraz,
        biezacy_obraz,
        poprzednie_punkty,
        None,
        **lk_params
    )

    return nowe_punkty, status, blad


# ============================================================
# TODO 4: Rysowanie aktualnych punktów
# ============================================================

def rysuj_punkty(obraz, punkty):
    """
    Rysuje aktualne pozycje śledzonych punktów.

    TODO:
    - Dla każdego punktu narysuj okrąg
    - Zwróć zmodyfikowany obraz
    """
    if punkty is None:
        return obraz

    for punkt in punkty:
        x, y = punkt.ravel()
        cv2.circle(obraz, (int(x), int(y)), 5, (255, 255, 255), -1)

    return obraz


# ============================================================
# TODO 5: Rysowanie trajektorii
# ============================================================

def rysuj_trajektorie(maska, stare_punkty, nowe_punkty):
    """
    Rysuje trajektorie punktów na osobnej masce.

    TODO:
    - Dla każdej pary punktów (stary, nowy) narysuj linię
    - Aktualizuj maskę trajektorii
    - Zwróć zaktualizowaną maskę
    """
    if stare_punkty is None or nowe_punkty is None:
        return maska

    for stary, nowy in zip(stare_punkty, nowe_punkty):
        x_stare, y_stare = stary.ravel()
        x_nowe, y_nowe = nowy.ravel()

        cv2.line(
            maska,
            (int(x_stare), int(y_stare)),
            (int(x_nowe), int(y_nowe)),
            (255, 255, 255),
            2
        )

    return maska


# ============================================================
# Główna pętla przetwarzania wideo
# ============================================================

def przetwarzaj_wideo(sciezka_wideo: str):
    cap = wczytaj_wideo(sciezka_wideo)

    # Wczytanie pierwszej klatki
    poprawnie, pierwsza_klatka = cap.read()
    if not poprawnie:
        print("BŁĄD: Nie można odczytać pierwszej klatki filmu", file=sys.stderr)
        return

    poprzedni_szary = cv2.cvtColor(pierwsza_klatka, cv2.COLOR_BGR2GRAY)

    # Wykrycie początkowych punktów
    poprzednie_punkty = wykryj_punkty(poprzedni_szary)

    # Maska do rysowania trajektorii
    maska_trajektorii = np.zeros_like(pierwsza_klatka)

    while True:
        poprawnie, klatka = cap.read()
        if not poprawnie:
            break

        biezacy_szary = cv2.cvtColor(klatka, cv2.COLOR_BGR2GRAY)

        # Śledzenie punktów
        nowe_punkty, status, blad = sledz_punkty(
            poprzedni_szary,
            biezacy_szary,
            poprzednie_punkty
        )

        # TODO 6:
        # - Odfiltruj tylko poprawnie śledzone punkty na podstawie status
        # - Obsłuż przypadek, gdy nie zostały żadne punkty
        # - Opcjonalnie: ponownie wykryj punkty, jeśli ich liczba jest zbyt mała

        # Odfiltrowanie tylko poprawnie śledzonych punktów
        if nowe_punkty is None or status is None:
            break

        dobre_nowe = nowe_punkty[status == 1]
        dobre_stare = poprzednie_punkty[status == 1]

        # Obsługa przypadku, gdy nie zostały żadne punkty
        if len(dobre_nowe) == 0:
            break

        # Przywrócenie formatu Nx1x2 zgodnego z OpenCV
        nowe_punkty = dobre_nowe.reshape(-1, 1, 2)
        poprzednie_punkty = dobre_stare.reshape(-1, 1, 2)

        # Rysowanie trajektorii
        maska_trajektorii = rysuj_trajektorie(
            maska_trajektorii,
            poprzednie_punkty,
            nowe_punkty
        )

        # Rysowanie aktualnych punktów
        wynik = rysuj_punkty(klatka.copy(), nowe_punkty)

        # Połączenie obrazu z maską trajektorii
        wynik = cv2.add(wynik, maska_trajektorii)

        # TODO 7:
        # - Dodaj opcjonalny napis z liczbą śledzonych punktów
        # - Dodaj opcjonalne informacje diagnostyczne

        liczba_punktow = 0 if nowe_punkty is None else len(nowe_punkty)

        cv2.putText(
            wynik,
            f"Liczba sledzonych punktow: {liczba_punktow}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            wynik,
            "q / ESC - wyjscie",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Optical Flow - sledzenie punktów", wynik)

        klawisz = cv2.waitKey(30) & 0xFF
        if klawisz in (ord("q"), 27):
            break

        poprzedni_szary = biezacy_szary.copy()
        poprzednie_punkty = nowe_punkty

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Funkcja główna
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LAB4 - Optical Flow, szablon startowy")
    parser.add_argument("--video", required=True, help="Ścieżka do pliku wideo")

    args = parser.parse_args()

    przetwarzaj_wideo(args.video)


if __name__ == "__main__":
    main()
