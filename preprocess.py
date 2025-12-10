# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import open3d as o3d
import numpy as np
import os

# --- AYARLAR ---
SOURCE = "./kaynak_veriler"      # Ham verilerin bulunduğu klasör
DEST = "./data/train"            # Preprocess sonrası verilerin gideceği klasör
POINTS = 4096                    # Her pakette kaç tane nokta olacak
BLOCK_SIZE = 1.5                 # Kesici boyutu (1.5 metre x 1.5 metre)
STRIDE = 0.5                     # Her kesimden sonra ne kadar kaydıracağız?

# Dosya -> Etiket (0:Arkaplan, 1:Elma)
FILES = {
    'Tree_1_0_LEAVES.ply': 0,
    'Tree_1_0_BRANCHES.ply': 0,
    'Tree_1_0_MAINSTEM.ply': 0,
    'Tree_1_0_APPLES.ply': 1
}




def main():
    os.makedirs(DEST, exist_ok=True)
    all_pts, all_lbls = [], []


    # Örnek: Elma dosyasından gelen bir nokta
    # [Koordinat X, Koordinat Y, Koordinat Z]  ---> [Koordinat X, Koordinat Y, Koordinat Z, ETİKET]
    # [-12.50,       5.30,        2.10      ]  ---> [-12.50,       5.30,        2.10,       1    ]


    # 1. Okuma ve Birleştirme
    print("Dosyalar okunuyor...")
    for fname, lbl_val in FILES.items():                                # Örnek: İlk turda fname = 'Tree_1_0_APPLES.ply' olur, lbl_val = 1 olur.
        path = os.path.join(SOURCE, fname)                              # Tam adres kontrolu
        if os.path.exists(path):                    
            pts = np.asarray(o3d.io.read_point_cloud(path).points)      #o3d.io.read_point_cloud(path): Dosyayı açar.
                                                                        #.points: Sadece X, Y, Z koordinatlarını alır (renkleri vs. atar).
                                                                        #np.asarray: Matrise çevirir.
                                                                        #Örnek çıktı: [-12.5, 5.0, 2.1]
                                                                        
            if len(pts) > 0:                                            #np.full komutu, 5000 satırlık ve sadece 1 rakamından oluşan yeni bir matris yaratır
                lbls = np.full((len(pts), 1), lbl_val)                  
                all_pts.append(pts)                                     
                all_lbls.append(lbls)
                
    if not all_pts: return print("Hata: Dosya bulunamadı.")
    
    data = np.hstack((np.concatenate(all_pts), np.concatenate(all_lbls)))       #Noktalar(X,Y,Z) ve Etiketler(0veya1) listelerini alıp yan yana yapıştırıyoruz.
    print(f"Toplam {len(data)} nokta birleştirildi. Parçalanıyor...")    #artık elimizde [N, 4] boyutunda devasa bir matris var.
                                                                                #ilk 3 sutun konum 4. sutun etiket
    # 2. Parçalama (Sliding Window)
    xyz = data[:, :3]
    min_l, max_l = np.min(xyz, 0), np.max(xyz, 0)                               #ağacın en uç noktalarını belirliyoruz
    count = 0

    x_range = np.arange(min_l[0], max_l[0] - BLOCK_SIZE, STRIDE)                #minden başlayıo maxa kadar gidiyoruz
    y_range = np.arange(min_l[1], max_l[1] - BLOCK_SIZE, STRIDE)                #stride belirlenen değerde ilerle

    for x in x_range:
        for y in y_range:
            # Kutunun içindekileri bul
            mask = (xyz[:,0]>=x) & (xyz[:,0]<x+BLOCK_SIZE) & (xyz[:,1]>=y) & (xyz[:,1]<y+BLOCK_SIZE)
            block = data[mask]
            
            if len(block) < 100: continue  # Eğer içeride 100den az nokta varsa burası boştur kaydetmeden geç

            # 4096 noktaya sabitle
            idx = np.random.choice(len(block), POINTS, replace=(len(block)<POINTS))
            sample = block[idx]

            # Kaydet
            np.save(f"{DEST}/sample_{count:04d}.npy", sample.astype(np.float32))
            count += 1

    print(f"Bitti. {count} adet parça oluşturuldu.")

if __name__ == "__main__":
    main()
