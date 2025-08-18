#!/usr/bin/env python3
# tests/benchmark_complete.py

import requests
import time
import json
import sys
import os
from PIL import Image
import io
from tabulate import tabulate

API_URL = os.environ.get('API_URL', 'http://localhost:5000')

def benchmark_model(model_id, image_path, num_runs=3):
    """Benchmark un modèle spécifique"""
    
    # Changer de modèle
    response = requests.post(f"{API_URL}/switch/{model_id}")
    if response.status_code != 200:
        print(f"❌ Erreur changement modèle {model_id}")
        return None
    
    # Charger l'image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    times = []
    inference_times = []
    
    for i in range(num_runs):
        buffer = io.BytesIO(image_bytes)
        files = {'image': ('test.png', buffer, 'image/png')}
        
        start = time.time()
        response = requests.post(f"{API_URL}/predict", files=files)
        total_time = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            times.append(total_time)
            inference_times.append(data['inference_time_ms'])
    
    return {
        'model': model_id,
        'avg_total': sum(times) / len(times),
        'avg_inference': sum(inference_times) / len(inference_times),
        'min_inference': min(inference_times),
        'max_inference': max(inference_times)
    }

def main():
    print("="*60)
    print("BENCHMARK COMPLET - SEGFORMER FP32 vs INT8")
    print("="*60)
    
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'test_images/test_000.png'
    print(f"Image de test : {image_path}")
    print(f"Nombre de runs par modèle : 3\n")
    
    models = [
        'segformer_b0_fp32', 'segformer_b0_int8',
        'segformer_b1_fp32', 'segformer_b1_int8',
        'segformer_b2_fp32', 'segformer_b2_int8'
    ]
    
    results = []
    
    for model_id in models:
        print(f"⏱️  Test {model_id}...")
        result = benchmark_model(model_id, image_path)
        if result:
            results.append(result)
    
    # Tableau comparatif
    print("\n" + "="*60)
    print("RÉSULTATS")
    print("="*60)
    
    # Formater pour affichage
    table_data = []
    for r in results:
        variant = r['model'].split('_')[1].upper()
        precision = r['model'].split('_')[2].upper()
        table_data.append([
            f"SegFormer-{variant}",
            precision,
            f"{r['avg_inference']:.1f}",
            f"{r['min_inference']:.1f}",
            f"{r['max_inference']:.1f}",
            f"{r['avg_total']:.1f}"
        ])
    
    headers = ["Modèle", "Précision", "Inférence (ms)", "Min", "Max", "Total (ms)"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Analyse comparative
    print("\n📊 ANALYSE")
    print("-"*40)
    
    for variant in ['b0', 'b1', 'b2']:
        fp32 = next((r for r in results if f'{variant}_fp32' in r['model']), None)
        int8 = next((r for r in results if f'{variant}_int8' in r['model']), None)
        
        if fp32 and int8:
            speedup = fp32['avg_inference'] / int8['avg_inference']
            diff_pct = (int8['avg_inference'] - fp32['avg_inference']) / fp32['avg_inference'] * 100
            
            print(f"\n{variant.upper()}:")
            print(f"  FP32 : {fp32['avg_inference']:.1f} ms")
            print(f"  INT8 : {int8['avg_inference']:.1f} ms")
            print(f"  Différence : {diff_pct:+.1f}%")
            
            if speedup > 1:
                print(f"  → INT8 {speedup:.2f}x plus rapide ✅")
            else:
                print(f"  → INT8 {1/speedup:.2f}x plus lent ⚠️")
    
    # Rappel des tailles
    print("\n💾 TAILLES DES MODÈLES")
    print("-"*40)
    sizes = {
        'b0': {'fp32': 14.5, 'int8': 4.6},
        'b1': {'fp32': 52.5, 'int8': 14.2},
        'b2': {'fp32': 104.9, 'int8': 28.3}
    }
    
    for variant, size in sizes.items():
        reduction = (1 - size['int8'] / size['fp32']) * 100
        print(f"{variant.upper()}: {size['fp32']:.1f} MB → {size['int8']:.1f} MB (réduction {reduction:.1f}%)")

if __name__ == "__main__":
    main()