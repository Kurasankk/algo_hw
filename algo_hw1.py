#!/usr/bin/env python3

import sys, math, random, argparse
from collections import Counter
from Bio import SeqIO

# Параметры команднойй строки
parser = argparse.ArgumentParser(description="HMM + Viterbi для химерных последовательностей")
parser.add_argument("--genome_low", required=True, help="FASTA низко-GC генома")
parser.add_argument("--genome_high", required=True, help="FASTA высоко-GC генома")
parser.add_argument("--chimera", action="store_true", help="Сгенерировать химерную последовательность")
parser.add_argument("--length", type=int, default=10000, help="Длина химерной последовательности")
parser.add_argument("--avg_fragment", type=int, default=300, help="Средняя длина фрагмента")
parser.add_argument("--input", help="Входная последовательность для декодирования")
parser.add_argument("--true_labels", help="Файл с истинными метками")
parser.add_argument("--output", help="Файл для сохранения предсказаний")
parser.add_argument("--verbose", action="store_true", help="Подробный вывод")
args = parser.parse_args()

# Функции
def read_fasta(file_path):
    """Чтение первой последовательности из FASTA"""
    records = list(SeqIO.parse(file_path, "fasta"))
    return str(records[0].seq).upper() if records else ""

def gc_content(seq):
    """Процент GC"""
    seq = [c for c in seq if c in 'ATGC']
    return sum(1 for c in seq if c in 'GC') / len(seq) if seq else 0.0

def nucleotide_probs(seq):
    """Эмиссионные вероятности для A,T,G,C с небольшим сглаживанием"""
    counts = Counter(c for c in seq if c in 'ATGC')
    total = sum(counts.values()) + 4  # сглаживание
    gc = (counts.get('G',0) + counts.get('C',0) + 2) / total
    at = 1 - gc
    return {'A': at/2, 'T': at/2, 'G': gc/2, 'C': gc/2}

def clean_fragment(seq, length):
    """Возвращает случайный чистый фрагмент без N"""
    max_start = len(seq) - length
    if max_start < 0: return None
    for _ in range(100):
        start = random.randint(0, max_start)
        frag = seq[start:start+length]
        if all(c in 'ATGC' for c in frag): return frag
    return None

def generate_chimera(seq_low, seq_high, total_len, avg_len):
    """Создание химерной последовательности с чередующимися фрагментами"""
    chimera, labels = [], []
    state = random.choice([1,2])
    total, frag_count = 0, 0
    while total < total_len:
        flen = max(50, int(random.expovariate(1/avg_len)))
        frag = clean_fragment(seq_low if state==1 else seq_high, flen)
        if not frag: continue
        chimera.append(frag)
        labels.extend([str(state)]*len(frag))
        total += len(frag)
        frag_count += 1
        state = 3 - state
    chimera = ''.join(chimera)[:total_len]
    labels = labels[:total_len]
    return chimera, labels

def viterbi(seq, emit_low, emit_high, mean_len):
    """Витерби с логарифмами вероятностей"""
    T = len(seq)
    states = 2
    stay = (mean_len-1)/mean_len      # вероятность остаться
    switch = 1/mean_len               # вероятность смены
    log_start = [math.log(0.5)]*states
    log_trans = [[math.log(stay), math.log(switch)],
                 [math.log(switch), math.log(stay)]]
    log_emit = [{n: math.log(emit_low[n]) for n in 'ATGC'},
                {n: math.log(emit_high[n]) for n in 'ATGC'}]
    # Инициализация
    v = [[0]*states for _ in range(T)]
    back = [[0]*states for _ in range(T)]
    v[0] = [log_start[i] + log_emit[i].get(seq[0], math.log(0.25)) for i in range(states)]
    # Прямой проход
    for t in range(1,T):
        for j in range(states):
            max_val, max_idx = max(
                ((v[t-1][i] + log_trans[i][j] + log_emit[j].get(seq[t], math.log(0.25)), i)
                 for i in range(states)), key=lambda x: x[0])
            v[t][j] = max_val
            back[t][j] = max_idx
    # Обратный проход
    pred = ['']*T
    best = 0 if v[T-1][0] > v[T-1][1] else 1
    pred[T-1] = str(best+1)
    for t in range(T-2, -1, -1):
        best = back[t+1][best]
        pred[t] = str(best+1)
    return pred

def accuracy(pred, true):
    return sum(p==t for p,t in zip(pred,true)) / len(true) * 100

# Основа
gen_low = read_fasta(args.genome_low)
gen_high = read_fasta(args.genome_high)
gc_low, gc_high = gc_content(gen_low), gc_content(gen_high)
if args.verbose:
    print(f"GC низкого генома: {gc_low:.3f}, GC высокого генома: {gc_high:.3f}")

# Генерация химерной последовательности
if args.chimera:
    chim_seq, true_labels = generate_chimera(gen_low, gen_high, args.length, args.avg_fragment)
    with open("chimera.fasta",'w') as f:
        f.write(f">chimera length={len(chim_seq)}\n")
        for i in range(0,len(chim_seq),80): f.write(chim_seq[i:i+80]+'\n')
    with open("chimera_true_labels.txt",'w') as f: f.write(''.join(true_labels)+'\n')
    if args.verbose:
        print(f"Сгенерирована химера длиной {len(chim_seq)} нуклеотидов, фрагментов: {len(true_labels)//args.avg_fragment}")

# Декодирование
if args.input:
    try:
        records = list(SeqIO.parse(args.input,"fasta"))
        input_seq = str(records[0].seq).upper()
    except:
        with open(args.input,'r') as f: input_seq = ''.join(line.strip().upper() for line in f if not line.startswith('>'))
    emit_low, emit_high = nucleotide_probs(gen_low), nucleotide_probs(gen_high)
    pred = viterbi(input_seq, emit_low, emit_high, args.avg_fragment)
    if args.output:
        with open(args.output,'w') as f: f.write(''.join(pred)+'\n')
        if args.verbose: print(f"Результаты сохранены в {args.output}")
    if args.true_labels:
        with open(args.true_labels,'r') as f: true_seq = f.read().strip()
        acc = accuracy(pred,true_seq)
        print(f"Точность предсказаний: {acc:.2f}%")

print("\nСделано")
