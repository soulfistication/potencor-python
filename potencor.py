#!/usr/bin/env python3
"""
potencor.py ? Python port of POTENCOR.PAS / potencor.cpp (EEG power & correlations).

Binary streams: little-endian int16 per sample (Turbo Pascal FILE OF integer).
"""

from __future__ import annotations

import math
import struct
import sys
from typing import List, Optional, TextIO, BinaryIO

MXPUNTOS = 1024
MAR = 8
MITAD = 512
MAXCAN = 16
MAXARCH = 190


def bell() -> None:
    sys.stdout.write("\a")
    sys.stdout.flush()


def pause_console() -> None:
    try:
        input()
    except EOFError:
        pass


def upcase_char(c: str) -> str:
    return c.upper() if c else c


def lr_fn(dato: float) -> float:
    if dato < 99.9999 and dato > 0.0001:
        return math.log(dato / (100.0 - dato))
    if dato >= 99.9999:
        return 99.99
    return -99.99


def rz_fn(dato: float) -> float:
    if dato < 1.0 and dato > -1.0:
        return 0.5 * math.log((1.0 + dato) / (1.0 - dato))
    if dato >= 1.0:
        return 99.99
    return -99.99


def _text_next_eof(f: TextIO) -> bool:
    pos = f.tell()
    line = f.readline()
    f.seek(pos)
    return line == ""


def _bin_peek_eof(f: BinaryIO) -> bool:
    pos = f.tell()
    b = f.read(1)
    if not b:
        return True
    f.seek(pos)
    return False


class Potencor:
    def __init__(self) -> None:
        self.pi = math.acos(-1.0)

        self.ijk = self.frent = 0
        self.dei = self.des = self.te1i = self.te1s = self.te2i = self.te2s = 0
        self.al1i = self.al1s = self.al2i = self.numcan = self.cual = 0
        self.al2s = self.be1i = self.be1s = self.be2i = self.be2s = 0
        self.parapro = self.numarch = 0
        self.Hz32 = self.numpts = self.puntos = self.nu = self.segm = 0
        self.frecmu = self.segprom = 0

        self.facdiv = 0.0
        self.procordel = self.procorte1 = self.procorte2 = 0.0
        self.procoral1 = self.procoral2 = 0.0
        self.procorbe1 = self.procorbe2 = self.procorfou = 0.0
        self.corfou = self.cordel = self.corte1 = self.corte2 = 0.0
        self.coral1 = self.coral2 = self.corbe1 = self.corbe2 = 0.0

        self.siind = False
        self.formatoentero = False
        self.mismoar = False
        self.terminar = False
        self.finarc = False
        self.marcas = False
        self.inter = False

        self.xreal = [0.0] * (MXPUNTOS + 1)
        self.ximag = [0.0] * (MXPUNTOS + 1)
        self.yreal = [0.0] * (MXPUNTOS + 1)
        self.yimag = [0.0] * (MXPUNTOS + 1)
        self.seno = [0.0] * (MXPUNTOS + 1)
        self.cose = [0.0] * (MXPUNTOS + 1)
        self.p = [0] * (MXPUNTOS + 1)

        self.espectrox = [0.0] * (MITAD + 1)
        self.espectroy = [0.0] * (MITAD + 1)
        self.especxyre = [0.0] * (MITAD + 1)
        self.espaux = [0.0] * (MITAD + 1)
        self.espauy = [0.0] * (MITAD + 1)
        self.espauxyre = [0.0] * (MITAD + 1)
        self.indabs = [0.0] * (MITAD + 1)
        self.indrel = [0.0] * (MITAD + 1)
        self.indcor = [0.0] * (MITAD + 1)

        self.fe1: Optional[BinaryIO] = None
        self.fe2: Optional[BinaryIO] = None
        self.ft1: Optional[TextIO] = None
        self.ft2: Optional[TextIO] = None
        self.fm: Optional[TextIO] = None

        self.f_abf: Optional[TextIO] = None
        self.f_ref: Optional[TextIO] = None
        self.f_erf: Optional[TextIO] = None
        self.f_arf: Optional[TextIO] = None
        self.f_alf: Optional[TextIO] = None
        self.f_rlf: Optional[TextIO] = None
        self.f_ezf: Optional[TextIO] = None
        self.f_azf: Optional[TextIO] = None
        self.f_aux: Optional[TextIO] = None
        self.f_rux: Optional[TextIO] = None
        self.f_abs: Optional[TextIO] = None
        self.f_rel: Optional[TextIO] = None
        self.f_ter: Optional[TextIO] = None
        self.f_tra: Optional[TextIO] = None

        self.auxnom = ""
        self.ax1 = ""
        self.ax2 = ""
        self.nomcom = ""
        self.nomaux = ""
        self.renglon = ""
        self.tipmarc = ""

        self.archiv1: List[str] = [""] * (MAXARCH + 1)
        self.archiv2: List[str] = [""] * (MAXARCH + 1)
        self.noment1: List[str] = [""] * (MAXCAN + 1)
        self.noment2: List[str] = [""] * (MAXCAN + 1)

    def sencos(self) -> None:
        fang = self.pi * 2.0 / float(self.puntos)
        for j in range(0, self.puntos):
            j1 = j
            ibit = 0
            for _ in range(1, self.nu + 1):
                j2 = j1 // 2
                ibit = ibit * 2 + (j1 - 2 * j2)
                self.p[j] = ibit
                j1 = j2
            self.seno[j] = math.sin(float(j) * fang)
            self.cose[j] = math.cos(float(j) * fang)

    def comun_fft(self, nu_local: int, puntos_local: int, xr: List[float], xi: List[float]) -> None:
        n2 = puntos_local // 2
        nu1 = nu_local - 1
        k = 0
        for _l in range(1, nu_local + 1):
            while True:
                for _i in range(1, n2 + 1):
                    ar = 1
                    if nu1 >= 1:
                        for _ki in range(1, nu1 + 1):
                            ar *= 2
                    ar = k // ar
                    c = self.cose[self.p[ar]]
                    s = self.seno[self.p[ar]]
                    k1 = k + 1
                    k1n2 = k1 + n2
                    treal = xr[k1n2] * c + xi[k1n2] * s
                    timag = xi[k1n2] * c - xr[k1n2] * s
                    xr[k1n2] = xr[k1] - treal
                    xi[k1n2] = xi[k1] - timag
                    xr[k1] = xr[k1] + treal
                    xi[k1] = xi[k1] + timag
                    k += 1
                k += n2
                if k >= puntos_local:
                    break
            k = 0
            nu1 -= 1
            n2 //= 2
        for k in range(1, puntos_local + 1):
            ar = k - 1
            i = self.p[ar] + 1
            if i >= k:
                treal = xr[k]
                timag = xi[k]
                xr[k] = xr[i]
                xi[k] = xi[i]
                xr[i] = treal
                xi[i] = timag

    def leer_datos(self, _puntos_local: int, xr: List[float], yr: List[float]) -> None:
        if self.formatoentero:
            assert self.fe1 is not None and self.fe2 is not None
            fe1, fe2 = self.fe1, self.fe2
            for i in range(1, self.numpts + 1):
                if i < self.numpts and (_bin_peek_eof(fe1) or _bin_peek_eof(fe2)):
                    self.terminar = True
                raw1 = fe1.read(2)
                raw2 = fe2.read(2)
                aux1 = struct.unpack("<h", raw1)[0] if len(raw1) == 2 else 0
                aux2 = struct.unpack("<h", raw2)[0] if len(raw2) == 2 else 0
                xr[i] = float(aux1)
                yr[i] = float(aux2)
        else:
            assert self.ft1 is not None and self.ft2 is not None
            for i in range(1, self.numpts + 1):
                if i < self.numpts and (_text_next_eof(self.ft1) or _text_next_eof(self.ft2)):
                    self.terminar = True
                la = self.ft1.readline()
                lb = self.ft2.readline()
                xr[i] = float(la.split()[0]) if la.strip() else 0.0
                yr[i] = float(lb.split()[0]) if lb.strip() else 0.0

        if not self.terminar and self.numpts == 128:
            j = self.numpts * 2
            for i in range(self.numpts, 1, -1):
                xr[j] = xr[i]
                yr[j] = yr[i]
                xr[j - 1] = (xr[i] + xr[i - 1]) / 2.0
                yr[j - 1] = (yr[i] + yr[i - 1]) / 2.0
                j -= 2
            xr[2] = xr[1]
            xr[1] = (xr[1] + xr[self.numpts * 2]) / 2.0
            yr[2] = yr[1]
            yr[1] = (yr[1] + yr[self.numpts * 2]) / 2.0

    def usuario(self) -> None:
        print(" " * MAR + "Nombre del primer archivo de  nombres      ? ", end="", flush=True)
        nomar1 = sys.stdin.readline().rstrip("\n")
        print(" " * MAR + "Nombre del segundo archivo de  nombres     ? ", end="", flush=True)
        nomar2 = sys.stdin.readline().rstrip("\n")
        self.mismoar = nomar1 == nomar2

        with open(nomar1, "r", encoding="utf-8", errors="replace") as f1, open(
            nomar2, "r", encoding="utf-8", errors="replace"
        ) as f2:
            i = 0
            while True:
                i += 1
                la = f1.readline()
                lb = f2.readline()
                self.archiv1[i] = la.rstrip("\n")
                self.archiv2[i] = lb.rstrip("\n")
                ax1 = "".join(
                    upcase_char(self.archiv1[i][j - 1]) if j <= len(self.archiv1[i]) else ""
                    for j in range(1, 7)
                )
                ax2 = "".join(
                    upcase_char(self.archiv2[i][j - 1]) if j <= len(self.archiv2[i]) else ""
                    for j in range(1, 7)
                )
                if ax1 != ax2:
                    print()
                    bell()
                    print(
                        f"{self.archiv1[i]} y {self.archiv2[i]} "
                        "no son de la misma condicion (6 primeros caracteres)",
                        end="",
                    )
                    pause_console()
                    print()
                if i >= MAXARCH:
                    self.numarch = i
                    break
                if la == "" or lb == "":
                    self.numarch = i
                    break
            if self.numarch == MAXARCH:
                ea = f1.readline()
                eb = f2.readline()
                if ea != "" or eb != "":
                    print()
                    bell()
                    print(f"Se procesaran los primeros {MAXARCH} archivos de los archivos de nombres")
                    print()

        print(" " * MAR + "Son archivos de [T]exto o de [E]nteros     ? ", end="", flush=True)
        tipo_in = sys.stdin.readline().rstrip("\n")
        self.formatoentero = bool(tipo_in and tipo_in[0].lower() == "e")

        while True:
            print(" " * MAR + "Existe archivo de marcas (S/N)             ? ", end="", flush=True)
            renglon = sys.stdin.readline().rstrip("\n")
            simarc = upcase_char(renglon[0]) if renglon else " "
            if simarc in ("S", "N"):
                break
        self.marcas = simarc == "S"

        while True:
            print(" " * MAR + "Puntos por segmento de EEG [128,256,512 o 1024] ? ", end="", flush=True)
            self.numpts = int(sys.stdin.readline().strip())
            if self.numpts in (128, 256, 512, 1024):
                break

        self.puntos = 256 if self.numpts == 128 else self.numpts
        if self.puntos == 256:
            self.nu = 8
        elif self.puntos == 512:
            self.nu = 9
        else:
            self.nu = 10

        print(" " * MAR + "Segmentos  para  los  promedios (0=todos)  ? ", end="", flush=True)
        self.parapro = int(sys.stdin.readline().strip())
        print(" " * MAR + "Frecuencia de muestreo (en Hertz)          ? ", end="", flush=True)
        self.frecmu = int(sys.stdin.readline().strip())
        if self.numpts == 128:
            self.frecmu *= 2

        self.siind = False
        if self.frecmu in (128, 256, 512, 1024):
            while True:
                print(" " * MAR + "Analisis por  [F]recuencia o por  [B]andas ? ", end="", flush=True)
                renglon = sys.stdin.readline().rstrip("\n")
                c = upcase_char(renglon[0]) if renglon else " "
                if c in ("F", "B"):
                    break
            fac = float(self.puntos) * 128.0 / (1024.0 * float(self.frecmu))
            if c == "F":
                self.siind = True
                self.frent = 16 if (self.puntos == 256 and self.frecmu == 512) else 25
                self.Hz32 = int(207 * fac)
            else:
                self.dei = int(12 * fac)
                self.des = int(31 * fac)
                self.te1i = int(32 * fac)
                self.te1s = int(47 * fac)
                self.te2i = int(48 * fac)
                self.te2s = int(63 * fac)
                self.al1i = int(64 * fac)
                self.al1s = int(80 * fac)
                self.al2i = int(81 * fac)
                self.al2s = int(104 * fac)
                self.be1i = int(105 * fac)
                self.be1s = int(152 * fac)
                self.be2i = int(153 * fac)
                self.be2s = int(207 * fac)
        else:
            print()
            pairs = [
                ("delta", "dei", "des"),
                ("teta1", "te1i", "te1s"),
                ("teta2", "te2i", "te2s"),
                ("alfa1", "al1i", "al1s"),
                ("alfa2", "al2i", "al2s"),
                ("beta1", "be1i", "be1s"),
                ("beta2", "be2i", "be2s"),
            ]
            for _label, a, b in pairs:
                print(" " * MAR + f"Lugar en que empieza {_label} ? ", end="", flush=True)
                setattr(self, a, int(sys.stdin.readline().strip()))
                print(" " * MAR + f"Lugar en que termina {_label} ? ", end="", flush=True)
                setattr(self, b, int(sys.stdin.readline().strip()))

        print()
        if not self.siind:
            if self.dei == 0:
                self.dei = 1
            if self.des == self.te1i:
                self.te1i += 1
            if self.te1s == self.al1i:
                self.te2i += 1
            if self.te2s == self.al1i:
                self.al1i += 1
            if self.al1s == self.al2i:
                self.al2i += 1
            if self.al2s == self.be1i:
                self.be1i += 1
            if self.be1s == self.be2i:
                self.be2i += 1
            self.Hz32 = self.be2s
            facfrec = (float(self.frecmu) / 2.0) / float(self.puntos // 2)
            pad = " " * (MAR * 2)
            print("Estan predeterminados los siguientes limites para las bandas del EEG.")
            print(f"{pad}delta:  {self.dei * facfrec:5.2f}  a  {self.des * facfrec:5.2f} Hz")
            print(f"{pad}teta1:  {self.te1i * facfrec:5.2f}  a  {self.te1s * facfrec:5.2f} Hz")
            print(f"{pad}teta2:  {self.te2i * facfrec:5.2f}  a  {self.te2s * facfrec:5.2f} Hz")
            print(f"{pad}a1:  {self.al1i * facfrec:5.2f}  a  {self.al1s * facfrec:5.2f} Hz")
            print(f"{pad}a2:  {self.al2i * facfrec:5.2f}  a  {self.al2s * facfrec:5.2f} Hz")
            print(f"{pad}b1:  {self.be1i * facfrec:5.2f}  a  {self.be1s * facfrec:5.2f} Hz")
            print(f"{pad}b2:  {self.be2i * facfrec:5.2f}  a  {self.be2s * facfrec:5.2f} Hz")
            print()

    def calcor(self, inf: int, sup: int, rxy: List[float], espex: List[float], espey: List[float], exyre: List[float]) -> None:
        fac1 = fac2 = 0.0
        rxy[0] = 0.0
        for i in range(inf, sup + 1):
            fac1 += espex[i]
        for i in range(inf, sup + 1):
            fac2 += espey[i]
        for i in range(inf, sup + 1):
            rxy[0] += exyre[i]
        if fac1 >= 1.0 and fac2 >= 1.0:
            rxy[0] /= math.sqrt(fac1 * fac2)
        else:
            rxy[0] = 0.0

    def imprime_pot(self, cual_local: int, espectro: List[float], noment: str) -> None:
        assert self.f_aux is not None and self.f_rux is not None
        if self.siind:
            if self.puntos == 256:
                if self.frecmu == 128:
                    j = 2
                    for i in range(1, self.frent + 1):
                        self.indabs[i] = espectro[j] + espectro[j - 1]
                        j += 2
                elif self.frecmu in (256, 512):
                    for i in range(1, self.frent + 1):
                        self.indabs[i] = espectro[i]
            if self.puntos == 512:
                if self.frecmu == 128:
                    j = 4
                    for i in range(1, self.frent + 1):
                        self.indabs[i] = (
                            espectro[j] + espectro[j - 1] + espectro[j - 2] + espectro[j - 3]
                        )
                        j += 4
                elif self.frecmu == 256:
                    j = 2
                    for i in range(1, self.frent + 1):
                        self.indabs[i] = espectro[j] + espectro[j - 1]
                        j += 2
                elif self.frecmu == 512:
                    for i in range(1, self.frent + 1):
                        self.indabs[i] = espectro[i]
            if self.puntos == 1024:
                if self.frecmu == 128:
                    j = 8
                    for i in range(1, self.frent + 1):
                        self.indabs[i] = (
                            espectro[j] + espectro[j - 1] + espectro[j - 2] + espectro[j - 3]
                        )
                        self.indabs[i] = (
                            espectro[j - 4] + espectro[j - 5] + espectro[j - 6] + espectro[j - 7]
                        )
                        j += 8
                elif self.frecmu == 256:
                    j = 4
                    for i in range(1, self.frent + 1):
                        self.indabs[i] = (
                            espectro[j] + espectro[j - 1] + espectro[j - 2] + espectro[j - 3]
                        )
                        j += 4
                elif self.frecmu == 512:
                    j = 2
                    for i in range(1, self.frent + 1):
                        self.indabs[i] = espectro[j] + espectro[j - 1]
                        j += 2
                elif self.frecmu == 1024:
                    for i in range(1, self.frent + 1):
                        self.indabs[i] = espectro[i]

            for i in range(1, self.frent + 1):
                self.indabs[i] *= self.facdiv
            stot = sum(self.indabs[i] for i in range(1, self.frent + 1))
            for i in range(1, self.frent + 1):
                self.indrel[i] = 100.0 * (self.indabs[i] / stot)

            if cual_local == 1:
                assert self.f_abf is not None and self.f_alf is not None and self.f_ref is not None and self.f_rlf is not None
                self.f_abf.write("".join(f"{self.indabs[i]:14.1f}" for i in range(1, self.frent + 1)))
                self.f_abf.write(f"  {noment} {self.segprom}\n")
                for i in range(1, self.frent + 1):
                    if self.indabs[i] < 1:
                        self.indabs[i] = 1
                self.f_alf.write("".join(f"{math.log(self.indabs[i]):7.2f}" for i in range(1, self.frent + 1)))
                self.f_alf.write(f"  {noment} {self.segprom}\n")
            else:
                self.f_aux.write("".join(f"{self.indabs[i]:14.1f}" for i in range(1, self.frent + 1)))
                self.f_aux.write(f"  {noment} {self.segprom}\n")
                for i in range(1, self.frent + 1):
                    if self.indabs[i] < 1:
                        self.indabs[i] = 1
                self.f_aux.write("".join(f"{math.log(self.indabs[i]):7.2f}" for i in range(1, self.frent + 1)))
                self.f_aux.write(f"  {noment} {self.segprom}\n")

            if cual_local == 1:
                self.f_ref.write("".join(f"{self.indrel[i]:7.2f}" for i in range(1, self.frent + 1)))
                self.f_ref.write(f"  {noment} {self.segprom}\n")
                self.f_rlf.write("".join(f"{lr_fn(self.indrel[i]):7.2f}" for i in range(1, self.frent + 1)))
                self.f_rlf.write(f"  {noment} {self.segprom}\n")
            else:
                self.f_rux.write("".join(f"{self.indrel[i]:7.2f}" for i in range(1, self.frent + 1)))
                self.f_rux.write(f"  {noment} {self.segprom}\n")
                self.f_rux.write("".join(f"{lr_fn(self.indrel[i]):7.2f}" for i in range(1, self.frent + 1)))
                self.f_rux.write(f"  {noment} {self.segprom}\n")
            return

        sud = sut1 = sut2 = sua1 = sua2 = sub1 = sub2 = 0.0
        for i in range(self.dei, self.des + 1):
            sud += espectro[i]
        for i in range(self.te1i, self.te1s + 1):
            sut1 += espectro[i]
        for i in range(self.te2i, self.te2s + 1):
            sut2 += espectro[i]
        for i in range(self.al1i, self.al1s + 1):
            sua1 += espectro[i]
        for i in range(self.al2i, self.al2s + 1):
            sua2 += espectro[i]
        for i in range(self.be1i, self.be1s + 1):
            sub1 += espectro[i]
        for i in range(self.be2i, self.be2s + 1):
            sub2 += espectro[i]

        sud *= self.facdiv
        sut1 *= self.facdiv
        sut2 *= self.facdiv
        sua1 *= self.facdiv
        sua2 *= self.facdiv
        sub1 *= self.facdiv
        sub2 *= self.facdiv
        stot = sud + sut1 + sut2 + sua1 + sua2 + sub1 + sub2
        pod = 100.0 * (sud / stot)
        pot1 = 100.0 * (sut1 / stot)
        pot2 = 100.0 * (sut2 / stot)
        poa1 = 100.0 * (sua1 / stot)
        poa2 = 100.0 * (sua2 / stot)
        pob1 = 100.0 * (sub1 / stot)
        pob2 = 100.0 * (sub2 / stot)

        assert self.f_abs is not None and self.f_rel is not None
        band_abs = (
            f"{sud:14.1f}{sut1:14.1f}{sut2:14.1f}{sua1:14.1f}{sua2:14.1f}"
            f"{sub1:14.1f}{sub2:14.1f}{stot:14.1f}"
        )
        if cual_local == 1:
            self.f_abs.write(band_abs)
        else:
            self.f_aux.write(band_abs)

        if sud < 1:
            sud = 1
        if sut1 < 1:
            sut1 = 1
        if sut2 < 1:
            sut2 = 1
        if sua1 < 1:
            sua1 = 1
        if sua2 < 1:
            sua2 = 1
        if sub1 < 1:
            sub1 = 1
        if sub2 < 1:
            sub2 = 1

        logline = (
            f"{math.log(sud):7.2f}{math.log(sut1):7.2f}{math.log(sut2):7.2f}{math.log(sua1):7.2f}"
            f"{math.log(sua2):7.2f}{math.log(sub1):7.2f}{math.log(sub2):7.2f}{math.log(stot):7.2f}"
            f"  {noment} {self.segprom}\n"
        )
        rel_line1 = (
            f"{pod:7.2f}{pot1:7.2f}{pot2:7.2f}{poa1:7.2f}{poa2:7.2f}{pob1:7.2f}{pob2:7.2f}"
        )
        rel_line2 = (
            f"{lr_fn(pod):7.2f}{lr_fn(pot1):7.2f}{lr_fn(pot2):7.2f}{lr_fn(poa1):7.2f}"
            f"{lr_fn(poa2):7.2f}{lr_fn(pob1):7.2f}{lr_fn(pob2):7.2f}  {noment} {self.segprom}\n"
        )
        if cual_local == 1:
            self.f_abs.write(logline)
            self.f_rel.write(rel_line1)
            self.f_rel.write(rel_line2)
        else:
            self.f_aux.write(logline)
            self.f_rux.write(rel_line1)
            self.f_rux.write(rel_line2)

    def imprime_cor(
        self,
        espex: List[float],
        espey: List[float],
        exyre: List[float],
        noment1: str,
        noment2: str,
    ) -> None:
        ini = fin = 0
        buf = [0.0]

        if self.finarc:
            if self.siind:
                for i in range(1, self.frent + 1):
                    self.indcor[i] /= float(self.segprom)
                if self.inter:
                    assert self.f_erf is not None and self.f_ezf is not None
                    self.f_erf.write("".join(f"{self.indcor[i]:7.2f}" for i in range(1, self.frent + 1)))
                    self.f_erf.write(f"  {noment1} {noment2} {self.segprom}\n")
                    self.f_ezf.write(
                        "".join(f"{rz_fn(self.indcor[i]):7.2f}" for i in range(1, self.frent + 1))
                    )
                    self.f_ezf.write(f"  {noment1} {noment2} {self.segprom}\n")
                else:
                    assert self.f_arf is not None and self.f_azf is not None
                    self.f_arf.write("".join(f"{self.indcor[i]:7.2f}" for i in range(1, self.frent + 1)))
                    self.f_arf.write(f"  {noment1} {noment2} {self.segprom}\n")
                    self.f_azf.write(
                        "".join(f"{rz_fn(self.indcor[i]):7.2f}" for i in range(1, self.frent + 1))
                    )
                    self.f_azf.write(f"  {noment1} {noment2} {self.segprom}\n")
            else:
                self.procorfou /= float(self.segprom)
                self.procordel /= float(self.segprom)
                self.procorte1 /= float(self.segprom)
                self.procorte2 /= float(self.segprom)
                self.procoral1 /= float(self.segprom)
                self.procoral2 /= float(self.segprom)
                self.procorbe1 /= float(self.segprom)
                self.procorbe2 /= float(self.segprom)
                assert self.f_ter is not None and self.f_tra is not None
                base = (
                    f"{self.procordel:7.3f}{self.procorte1:7.3f}{self.procorte2:7.3f}"
                    f"{self.procoral1:7.3f}{self.procoral2:7.3f}{self.procorbe1:7.3f}"
                    f"{self.procorbe2:7.3f}{self.procorfou:7.3f}"
                    f"{rz_fn(self.procordel):7.2f}{rz_fn(self.procorte1):7.2f}{rz_fn(self.procorte2):7.2f}"
                    f"{rz_fn(self.procoral1):7.2f}{rz_fn(self.procoral2):7.2f}{rz_fn(self.procorbe1):7.2f}"
                    f"{rz_fn(self.procorbe2):7.2f}{rz_fn(self.procorfou):7.2f}  {noment1} {noment2} {self.segprom}\n"
                )
                if self.inter:
                    self.f_ter.write(base)
                else:
                    self.f_tra.write(base)
            return

        if self.siind:
            for i in range(1, self.frent + 1):
                if self.puntos == 256:
                    if self.frecmu == 128:
                        ini, fin = 2 * (i - 1) + 1, 2 * i
                    elif self.frecmu in (256, 512):
                        ini = fin = i
                if self.puntos == 512:
                    if self.frecmu == 128:
                        ini, fin = 4 * (i - 1) + 1, 4 * i
                    elif self.frecmu == 256:
                        ini, fin = 2 * (i - 1) + 1, 2 * i
                    elif self.frecmu == 512:
                        ini = fin = i
                if self.puntos == 1024:
                    if self.frecmu == 128:
                        ini, fin = 8 * (i - 1) + 1, 8 * i
                    elif self.frecmu == 256:
                        ini, fin = 4 * (i - 1) + 1, 4 * i
                    elif self.frecmu == 512:
                        ini, fin = 2 * (i - 1) + 1, 2 * i
                    elif self.frecmu == 1024:
                        ini = fin = i
                self.calcor(ini, fin, buf, espex, espey, exyre)
                self.indcor[i] += buf[0]
        else:
            self.calcor(self.dei, self.des, buf, espex, espey, exyre)
            self.cordel = buf[0]
            self.calcor(self.te1i, self.te1s, buf, espex, espey, exyre)
            self.corte1 = buf[0]
            self.calcor(self.te2i, self.te2s, buf, espex, espey, exyre)
            self.corte2 = buf[0]
            self.calcor(self.al1i, self.al1s, buf, espex, espey, exyre)
            self.coral1 = buf[0]
            self.calcor(self.al2i, self.al2s, buf, espex, espey, exyre)
            self.coral2 = buf[0]
            self.calcor(self.be1i, self.be1s, buf, espex, espey, exyre)
            self.corbe1 = buf[0]
            self.calcor(self.be2i, self.be2s, buf, espex, espey, exyre)
            self.corbe2 = buf[0]
            self.calcor(self.dei, self.be2s, buf, espex, espey, exyre)
            self.corfou = buf[0]
            self.procorfou += self.corfou
            self.procordel += self.cordel
            self.procorte1 += self.corte1
            self.procorte2 += self.corte2
            self.procoral1 += self.coral1
            self.procoral2 += self.coral2
            self.procorbe1 += self.corbe1
            self.procorbe2 += self.corbe2

    def _close_data_files(self) -> None:
        if self.formatoentero:
            if self.fe1:
                self.fe1.close()
            if self.fe2:
                self.fe2.close()
            self.fe1 = self.fe2 = None
        else:
            if self.ft1:
                self.ft1.close()
            if self.ft2:
                self.ft2.close()
            self.ft1 = self.ft2 = None

    def _open_pair(self, path1: str, path2: str) -> None:
        if self.formatoentero:
            self.fe1 = open(path1, "rb")
            self.fe2 = open(path2, "rb")
        else:
            self.ft1 = open(path1, "r", encoding="utf-8", errors="replace")
            self.ft2 = open(path2, "r", encoding="utf-8", errors="replace")

    def _merge_aux(self) -> None:
        assert self.f_aux is not None and self.f_rux is not None
        self.f_aux.flush()
        self.f_rux.flush()
        self.f_aux.seek(0)
        self.f_rux.seek(0)
        if self.siind:
            assert self.f_abf is not None and self.f_alf is not None
            assert self.f_ref is not None and self.f_rlf is not None
            while True:
                line_a = self.f_aux.readline()
                if not line_a:
                    break
                tok = line_a.split()
                for i in range(1, self.frent + 1):
                    self.indabs[i] = float(tok[i - 1])
                tail_out = "  " + " ".join(tok[self.frent :]) + "\n"
                self.f_abf.write(
                    "".join(f"{self.indabs[i]:14.1f}" for i in range(1, self.frent + 1)) + tail_out
                )
                line_log = self.f_aux.readline()
                if not line_log:
                    break
                self.f_alf.write(line_log)
            while True:
                ln = self.f_rux.readline()
                if not ln:
                    break
                self.f_ref.write(ln)
                ln2 = self.f_rux.readline()
                if not ln2:
                    break
                self.f_rlf.write(ln2)
        else:
            assert self.f_abs is not None and self.f_rel is not None
            while True:
                ln = self.f_aux.readline()
                if not ln:
                    break
                self.f_abs.write(ln)
            while True:
                ln = self.f_rux.readline()
                if not ln:
                    break
                self.f_rel.write(ln)

        self.f_aux.close()
        self.f_rux.close()
        open("BORRAR1.TMP", "w", encoding="utf-8").close()
        open("BORRAR2.TMP", "w", encoding="utf-8").close()
        self.f_aux = open("BORRAR1.TMP", "r+", encoding="utf-8")
        self.f_rux = open("BORRAR2.TMP", "r+", encoding="utf-8")

    def _marcas_open(self, nom_row: str, nom_other: str) -> None:
        j = 0
        self.auxnom = ""
        while True:
            j += 1
            if j <= len(nom_row):
                self.auxnom += nom_row[j - 1]
            if j <= len(nom_row) and j <= len(nom_other):
                if upcase_char(nom_row[j - 1]) != upcase_char(nom_other[j - 1]):
                    bell()
                    print(f"El archivo de marcas debe ser el mismo para {nom_row} y {nom_other}", end="")
                    pause_console()
            if j >= len(nom_row) or (j < len(nom_row) and nom_row[j] == ".") or j >= 6:
                break
        self.fm = open(self.auxnom + ".RAM", "r", encoding="utf-8", errors="replace")

    def _segment_inner(self, nom1: str, nom2: str, do_cross: bool, write_pot: bool) -> None:
        while True:
            self.finarc = False
            self.segprom = 0
            self.terminar = False
            self.segm = 0
            self.procorfou = 0.0
            for i in range(1, self.frent + 1):
                self.indcor[i] = 0.0
            self.procordel = self.procorte1 = self.procorte2 = self.procoral1 = 0.0
            self.procoral2 = 0.0
            self.procorbe1 = self.procorbe2 = 0.0
            for i in range(0, self.Hz32 + 1):
                self.espectrox[i] = self.espectroy[i] = self.especxyre[i] = 0.0

            while True:
                self.leer_datos(self.puntos, self.xreal, self.yreal)
                if not self.terminar:
                    if self.marcas:
                        assert self.fm is not None
                        self.renglon = self.fm.readline().rstrip("\n")
                        self.tipmarc = upcase_char(self.renglon[0]) if self.renglon else " "
                    else:
                        self.tipmarc = "A"
                    self.segm += 1

                if not self.terminar and self.tipmarc == "A":
                    self.segprom += 1
                    for i in range(1, self.puntos + 1):
                        self.ximag[i] = self.yimag[i] = 0.0
                    self.comun_fft(self.nu, self.puntos, self.xreal, self.ximag)
                    for i in range(1, self.Hz32 + 1):
                        self.espaux[i] = self.xreal[i + 1] ** 2 + self.ximag[i + 1] ** 2
                        self.espectrox[i] += self.espaux[i]
                    if do_cross:
                        self.comun_fft(self.nu, self.puntos, self.yreal, self.yimag)
                        for i in range(1, self.Hz32 + 1):
                            self.espauy[i] = self.yreal[i + 1] ** 2 + self.yimag[i + 1] ** 2
                            self.espectroy[i] += self.espauy[i]
                        for i in range(1, self.Hz32 + 1):
                            self.espauxyre[i] = (
                                self.xreal[i + 1] * self.yreal[i + 1]
                                + self.ximag[i + 1] * self.yimag[i + 1]
                            )
                            self.especxyre[i] += self.espauxyre[i]
                        self.imprime_cor(self.espaux, self.espauy, self.espauxyre, nom1, nom2)

                if self.formatoentero:
                    assert self.fe1 is not None and self.fe2 is not None
                    if _bin_peek_eof(self.fe1) or _bin_peek_eof(self.fe2):
                        self.terminar = True
                else:
                    assert self.ft1 is not None and self.ft2 is not None
                    if _text_next_eof(self.ft1) or _text_next_eof(self.ft2):
                        self.terminar = True

                if self.terminar or (self.parapro != 0 and self.segm == self.parapro):
                    break

            if self.segprom > 0:
                for i in range(1, self.Hz32 + 1):
                    self.espectrox[i] /= float(self.segprom)
                    self.espectroy[i] /= float(self.segprom)
                    self.especxyre[i] /= float(self.segprom)
                if write_pot:
                    self.cual = 1
                    self.imprime_pot(self.cual, self.espectrox, nom1)
                    if do_cross:
                        self.cual = 2
                        self.imprime_pot(self.cual, self.espectroy, nom2)
                        self.finarc = True
                        self.imprime_cor(self.espectrox, self.espectroy, self.especxyre, nom1, nom2)
                else:
                    self.finarc = True
                    self.imprime_cor(self.espectrox, self.espectroy, self.especxyre, nom1, nom2)

            if self.terminar:
                break

    def run(self) -> None:
        for _ in range(24):
            print()
        self.usuario()
        self.sencos()
        self.facdiv = 4.0 / float(self.puntos * self.puntos)
        print()
        print(" " * MAR + "Nombre comun de los archivos de salida (max. 6 car.) ? ", end="", flush=True)
        self.nomaux = sys.stdin.readline().rstrip("\n")
        print()
        self.nomcom = ""
        for j, ch in enumerate(self.nomaux[:6]):
            if ch == ".":
                break
            self.nomcom += ch

        open("BORRAR1.TMP", "w", encoding="utf-8").close()
        open("BORRAR2.TMP", "w", encoding="utf-8").close()
        self.f_aux = open("BORRAR1.TMP", "r+", encoding="utf-8")
        self.f_rux = open("BORRAR2.TMP", "r+", encoding="utf-8")

        if self.siind:
            self.f_abf = open(self.nomcom + ".ABF", "w", encoding="utf-8")
            self.f_ref = open(self.nomcom + ".REF", "w", encoding="utf-8")
            self.f_erf = open(self.nomcom + ".ERF", "w", encoding="utf-8")
            self.f_arf = open(self.nomcom + ".ARF", "w", encoding="utf-8")
            self.f_alf = open(self.nomcom + ".ALF", "w", encoding="utf-8")
            self.f_rlf = open(self.nomcom + ".RLF", "w", encoding="utf-8")
            self.f_ezf = open(self.nomcom + ".EZF", "w", encoding="utf-8")
            self.f_azf = open(self.nomcom + ".AZF", "w", encoding="utf-8")
        else:
            self.f_abs = open(self.nomcom + ".ABS", "w", encoding="utf-8")
            self.f_rel = open(self.nomcom + ".REL", "w", encoding="utf-8")
            self.f_ter = open(self.nomcom + ".TER", "w", encoding="utf-8")
            self.f_tra = open(self.nomcom + ".TRA", "w", encoding="utf-8")

        self.ijk = 0
        while self.ijk < self.numarch:
            print()
            self.ijk += 1
            i = 1
            self.noment1[i] = self.archiv1[self.ijk]
            self.noment2[i] = self.archiv2[self.ijk]
            ax1 = "".join(
                upcase_char(self.noment1[i][k - 1]) if k <= len(self.noment1[i]) else ""
                for k in range(1, 7)
            )
            while self.ijk < self.numarch and i < MAXCAN:
                ax2 = "".join(
                    upcase_char(self.archiv1[self.ijk + 1][k - 1])
                    if k <= len(self.archiv1[self.ijk + 1])
                    else ""
                    for k in range(1, 7)
                )
                if ax1 != ax2:
                    break
                i += 1
                self.ijk += 1
                self.noment1[i] = self.archiv1[self.ijk]
                self.noment2[i] = self.archiv2[self.ijk]
            self.numcan = i

            self.inter = True
            for ii in range(1, self.numcan + 1):
                print(" " * (MAR * 2) + f"{self.noment1[ii]}   {self.noment2[ii]}")
                if self.marcas:
                    self._marcas_open(self.noment1[ii], self.noment2[ii])
                self._open_pair(self.noment1[ii], self.noment2[ii])
                self._segment_inner(
                    self.noment1[ii],
                    self.noment2[ii],
                    do_cross=not self.mismoar,
                    write_pot=True,
                )
                if self.marcas:
                    assert self.fm is not None
                    self.fm.close()
                    self.fm = None
                self._close_data_files()
                self._merge_aux()

            self.inter = False
            if self.numcan > 1:
                for ii in range(1, self.numcan):
                    for jj in range(ii + 1, self.numcan + 1):
                        print(" " * (MAR * 2) + f"{self.noment1[ii]}   {self.noment1[jj]}")
                        if self.marcas:
                            self._marcas_open(self.noment1[ii], self.noment1[jj])
                        self._open_pair(self.noment1[ii], self.noment1[jj])
                        self._segment_inner(
                            self.noment1[ii],
                            self.noment1[jj],
                            do_cross=True,
                            write_pot=False,
                        )
                        if self.marcas:
                            assert self.fm is not None
                            self.fm.close()
                            self.fm = None
                        self._close_data_files()

                if not self.mismoar:
                    for ii in range(1, self.numcan):
                        for jj in range(ii + 1, self.numcan + 1):
                            print(" " * (MAR * 2) + f"{self.noment2[ii]}   {self.noment2[jj]}")
                            if self.marcas:
                                self._marcas_open(self.noment2[ii], self.noment2[jj])
                            self._open_pair(self.noment2[ii], self.noment2[jj])
                            self._segment_inner(
                                self.noment2[ii],
                                self.noment2[jj],
                                do_cross=True,
                                write_pot=False,
                            )
                            if self.marcas:
                                assert self.fm is not None
                                self.fm.close()
                                self.fm = None
                            self._close_data_files()

        print()
        assert self.f_aux is not None and self.f_rux is not None
        self.f_aux.close()
        self.f_rux.close()
        if self.siind:
            assert self.f_abf is not None
            self.f_abf.close()
            assert self.f_ref is not None
            self.f_ref.close()
            assert self.f_erf is not None
            self.f_erf.close()
            assert self.f_arf is not None
            self.f_arf.close()
            assert self.f_alf is not None
            self.f_alf.close()
            assert self.f_rlf is not None
            self.f_rlf.close()
            assert self.f_ezf is not None
            self.f_ezf.close()
            assert self.f_azf is not None
            self.f_azf.close()
        else:
            assert self.f_ter is not None
            self.f_ter.close()
            assert self.f_tra is not None
            self.f_tra.close()
            assert self.f_abs is not None
            self.f_abs.close()
            assert self.f_rel is not None
            self.f_rel.close()
        bell()
        print("  ****    PROGRAMA TERMINADO    ****")


def main() -> None:
    Potencor().run()


if __name__ == "__main__":
    main()
