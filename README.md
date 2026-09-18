# BlenderProc Sentetik Küp Veri Üreticisi

BlenderProc 2.x ve Blender 3.x/4.x ile, robot kamerası koşullarına benzer 640×640 sentetik görüntüler üretir. Yalnızca küpler etiketlenir; saha çizgileri, bariyerler, direkler, kablolar, silindirler, robot parçaları ve gölge nesneleri arka plan sınıfında kalır.

Üretici aynı kare için eş zamanlı olarak şunları oluşturur:

- YOLO detection etiketleri
- COCO detection annotation dosyası
- Kamera, ışık, zemin, materyal, küp dönüşleri ve post-processing metadata'sı
- Tek worker veya paralel worker çıktıları
- `dataset.yaml`

## Öne çıkan özellikler

- Kamera yüksekliği, uzaklığı, azimuth, elevation, roll ve 24–45 mm odak uzaklığı çeşitliliği
- HDR/EXR ve JPG/JPEG/PNG dosyalarını tek havuzda birleştirip dosya bazında rastgele arka plan seçimi
- Son altı karede kullanılan arka plan dosyasını tekrar seçmeyen diversity-window
- 1–3 adet POINT/AREA/SPOT ışık; renk sıcaklığı, enerji, konum ve gölge yumuşaklığı randomizasyonu
- HDRI seçimi, dönüşü, strength ve renk sıcaklığı değişimi
- Renk/doku/roughness/yansıma çeşitliliğine sahip saha zemini
- %40 düzgün, %40 küçük eğimli, %20 belirgin eğimli veya farklı yüzü üzerine yatmış küpler
- Döndürülmüş küpün en alt noktasını zemine veya alttaki küpe oturtan yerleşim hesabı
- %30 düz renkli; geri kalanı desenli/FDM yüzeyli küpler
- Çizgili, ızgaralı, noktalı, geometrik, benekli, mermerimsi, iki renkli, gradyan, aşınmış ve baskı hatalı texture'lar
- Parlaklık, kontrast, gamma, renk sıcaklığı, exposure, Gaussian/motion blur, lens distortion ve sensör gürültüsü
- 160/240/320/480 px'e küçültüp yeniden 640×640'a büyüten düşük çözünürlük simülasyonu
- Nearest, bilinear veya bicubic yeniden örnekleme ve kontrollü JPEG artefaktları
- Cycles için `fast`, `balanced` ve `realistic` kalite profilleri
- CUDA, OptiX ve CPU+GPU hibrit desteği; adaptive sampling, denoiser, AgX/Filmic fallback
- Worker bazlı türetilmiş seed ile tekrar üretilebilir üretim

## Gereksinimler

- Python 3.11+
- BlenderProc 2.8+
- Blender 3.x veya 4.x
- NumPy, SciPy ve Pillow
- CUDA/OptiX uyumlu NVIDIA GPU isteğe bağlıdır

Kurulum:

```bash
uv sync
```

veya:

```bash
python -m pip install blenderproc numpy scipy pillow
```

## Proje yapısı

```text
.
├── generate.py             Ana BlenderProc üreticisi ve merkezi CONFIG
├── image_effects.py        Blender gerektirmeyen kamera/post-processing işlemleri
├── texture.py              FDM albedo + normal map ve desen üreticisi
├── gen_textures.py         Texture havuzunu önceden üretir
├── merge.py                Paralel YOLO/COCO çıktılarını birleştirir ve split yapar
├── draw_bbox.py            En az 10 veya tüm görüntüler için bbox önizlemesi
├── validate_dataset.py     YOLO/COCO/metadata sözleşmesini doğrular
├── tests/                  Blender gerektirmeyen testler
├── backgrounds/            HDR/EXR arka planlar
├── textures/               JPG/JPEG/PNG arka plan görselleri
├── fdm_textures/           Otomatik texture cache'i
└── output/
    ├── yolo/
    │   ├── images/
    │   ├── labels/
    │   └── metadata/
    ├── coco/
    │   ├── images/
    │   └── coco_annotations.json
    ├── workers/worker_N/   Paralel moddaki aynı YOLO/COCO yapısı
    ├── bbox_debug/
    ├── dataset.yaml
    └── errors.log
```

`.venv` okunmaz, değiştirilmez ve dağıtım paketine alınmaz.

## Tek kaynaklı EXR ve görsel arka plan havuzu

Dosyaları tercihen aşağıdaki gibi yerleştirin:

```text
backgrounds/
├── salon_01.exr
├── salon_02.hdr
└── saha_03.exr

textures/
├── zemin_01.jpg
├── zemin_02.jpeg
└── zemin_03.png
```

Her karede ortak havuzdan doğrudan bir dosya seçilir:

- `backgrounds/` ve `textures/` içindeki tüm HDR/EXR/JPG/JPEG/PNG dosyaları tek havuzda birleştirilir.
- Önce kaynak türü seçilmez; ortak havuzdaki her dosyanın seçilme şansı eşittir. Örneğin 80 EXR ve 20 PNG varsa uzun vadede seçimlerin yaklaşık `%80`i EXR, `%20`si PNG olur.
- Her iki klasör alt klasörleriyle birlikte taranır; `.PNG`, `.JPG`, `.EXR` gibi büyük/küçük harfli uzantılar desteklenir. İstenirse iki tür de bu klasörlerden herhangi birinde tutulabilir.
- Havuzda yalnızca tek dosya türü varsa doğal olarak bütün seçimler o türden yapılır.
- `BACKGROUND_DIVERSITY_WINDOW=6`, yeterli dosya bulunduğunda son altı seçimi tekrar ettirmez.
- HDR/EXR seçilirse yalnızca o dosya Blender world environment olarak kullanılır. Cycles destekliyorsa görünmeyen bir shadow catcher temas gölgelerini korur; başka görsel zemin kullanılmaz.
- JPG/JPEG/PNG seçilirse yalnızca o görüntü kameraya dönük ve görüş alanını tamamen dolduran arka plan düzlemine bağlanır. World nötr renkte kalır ve sahne fiziksel ışıklarla aydınlatılır; EXR/HDR yüklenmez.
- Aynı karede raster görsel ile HDRI hiçbir zaman birlikte arka plan olarak kullanılmaz. Ortak havuzdan seçilen tek dosya kullanılır.
- Seçilen kaynak türü ve dosya adı her kareye ait metadata içindeki `background.source_type` ve `background.name` alanlarına yazılır.
- HDRI karelerinde gerçekten Blender world node'una bağlanan dosya ayrıca `background.applied_name` alanına yazılır ve `validate_dataset.py` tarafından seçilen dosyayla karşılaştırılır.

`backgrounds/` ve `textures/` arka plan arama kökleridir; `fdm_textures/` ise yalnızca küp kaplama cache'ini içerir.
Program başlarken bulduğu HDRI ve JPG/PNG sayılarını terminale yazar. Küçük bir testte yalnızca `"hdri": 10` görülmesi, JPG/PNG havuzunun boş olmasından veya ortak havuzdaki dosya dağılımından kaynaklanabilir; kesin kontrol için başlangıçtaki dosya sayılarına bakın.
`validate_dataset.py`, bütün kareler tek arka plan türünden geldiyse bunu `warnings` alanında ayrıca bildirir.

## Merkezi varsayılan ayarlar

Tüm sahne ve görüntü çeşitliliği ayarları `generate.py` başındaki CONFIG bölümündedir.

| Ayar | Varsayılan | Açıklama |
|---|---:|---|
| `NUM_IMAGES` | `1000` | Toplam görüntü |
| `IMG_RESOLUTION` | `640` | Render ve son çıktı boyutu |
| `QUALITY_PROFILE` | `balanced` | Cycles profili |
| `RANDOM_SEED` | `42` | Ana tekrar üretilebilirlik seed'i |
| `MIN_CUBES` / `MAX_CUBES` | `2` / `12` | Kare başına küp sayısı |
| `CUBE_ORIENTATION_WEIGHTS` | `0.40/0.40/0.20` | Düzgün/küçük/belirgin eğim |
| `SOLID_COLOR_PROBABILITY` | `0.30` | Düz renkli küp olasılığı |
| `ENVIRONMENT_PROBABILITY` | `0.72` | En az bir çevre türü denenme olasılığı |
| `BACKGROUND_DIVERSITY_WINDOW` | `6` | Yakın karelerde tekrar engelleme penceresi |
| `IMAGE_BACKGROUND_DISTANCE` | `30.0` | Kamera düzleminin güvenli arka plan uzaklığı |
| `ENV_OBJECT_COUNT_RANGE` | `1–7` | Çevre nesnesi sayısı |
| `LOW_RES_PROBABILITY` | `0.35` | Düşük çözünürlük simülasyonu |
| `LOW_RES_SIZES` | `160,240,320,480` | Küçültme uzun kenarları |
| `JPEG_COMPRESSION_PROBABILITY` | `0.25` | JPEG artefakt olasılığı |
| `JPEG_QUALITY_RANGE` | `58–90` | JPEG kalite aralığı |
| `MIN_VISIBILITY_RATIO` | `0.40` | Annotation için görünürlük eşiği |

Düz renk paleti: beyaz, siyah, gri, kırmızı, mavi, yeşil, sarı, turuncu, mor, pembe ve kahverengi. Düz materyallerde mikro bump, roughness, specular ve IOR varyasyonu bulunur.

Çevre türlerinin her birinin bulunma olasılığı, renk listesi, genel boyut ve türe özgü ölçü aralıkları CONFIG bölümünden değiştirilebilir. Çevre nesneleri `category_id=0`, küpler `category_id=1` olarak atanır; YOLO/COCO yazıcıları sadece küpleri kabul eder.

## Kalite profilleri

| Profil | Sample | Noise threshold | Max bounce | Kullanım |
|---|---:|---:|---:|---|
| `fast` | 32 | 0.030 | 4 | Hızlı deneme ve büyük hacim |
| `balanced` | 64 | 0.010 | 6 | Varsayılan, dengeli üretim |
| `realistic` | 192 | 0.005 | 8 | Final veri, daha yavaş |

Profiller adaptive sampling, diffuse/glossy/transmission/transparent bounce, firefly clamp ve glossy filter değerlerini birlikte ayarlar. `--samples` verilirse profilin sample değeri ezilir; diğer profil ayarları korunur.

## CLI seçenekleri

BlenderProc argümanlarının önündeki `--` ayıracını koruyun.

| Seçenek | Açıklama |
|---|---|
| `--quality-profile fast|balanced|realistic` | Cycles kalite profili |
| `--optix` | OptiX GPU |
| `--gpu` | CUDA GPU |
| `--hybrid` | CUDA CPU+GPU hibrit |
| `--eevee` | Eevee hızlı render |
| `--num-images N` | Toplam görüntü sayısı |
| `--samples N` | Sample sayısını ezer |
| `--seed N` | Ana seed |
| `--worker-id N` | Worker index'i, 0'dan başlar |
| `--num-workers N` | Toplam worker |
| `--texture-pool-size N` | FDM texture havuzu |
| `--regen-textures` | Texture cache'ini yeniden üretir |
| `--gen-textures` | Sadece texture üretir |
| `--low-res-probability P` | `0.0–1.0` düşük çözünürlük olasılığı |
| `--low-res-sizes 160,240,320,480` | Virgülle ayrılmış uzun kenarlar |
| `--jpeg-compression-probability P` | `0.0–1.0` JPEG olasılığı |
| `--jpeg-quality-range 58,90` | Virgüllü veya iki ayrı sayı |
| `--test` | 10 kare, en fazla 16 sample ve 10 texture |

## Linux komutları

Texture havuzu:

```bash
python gen_textures.py --pool-size 200 --seed 42
```

10 karelik hızlı test:

```bash
blenderproc run generate.py -- --test --quality-profile fast --seed 42
python draw_bbox.py --limit 10
python validate_dataset.py
```

Tek worker, dengeli OptiX:

```bash
blenderproc run generate.py -- \
  --optix --quality-profile balanced --num-images 1000 --seed 42 \
  --low-res-probability 0.35 --low-res-sizes 160,240,320,480 \
  --jpeg-compression-probability 0.25 --jpeg-quality-range 58,90
```

Dört paralel worker:

```bash
for i in 0 1 2 3; do
  blenderproc run generate.py -- \
    --optix --quality-profile balanced --num-images 1000 \
    --num-workers 4 --worker-id "$i" --seed 42 &
done
wait
python merge.py
python draw_bbox.py --limit 10
python validate_dataset.py
```

## Windows PowerShell komutları

Aşağıdaki örnek `.venv` içindeki BlenderProc'u kullanır. Global kurulum varsa `$BlenderProc = "blenderproc"` yapabilirsiniz.

```powershell
$BlenderProc = ".\.venv\Scripts\blenderproc.exe"
python .\gen_textures.py --pool-size 200 --seed 42
```

10 karelik hızlı test:

```powershell
& $BlenderProc run .\generate.py -- --test --quality-profile fast --seed 42
python .\draw_bbox.py --limit 10
python .\validate_dataset.py
```

Tek worker, dengeli OptiX:

```powershell
& $BlenderProc run .\generate.py -- `
  --optix --quality-profile balanced --num-images 1000 --seed 42 `
  --low-res-probability 0.35 --low-res-sizes 160,240,320,480 `
  --jpeg-compression-probability 0.25 --jpeg-quality-range 58,90
```

Dört paralel worker:

```powershell
$jobs = 0..3 | ForEach-Object {
    $worker = $_
    Start-Process -FilePath $BlenderProc -PassThru -ArgumentList @(
        "run", ".\generate.py", "--",
        "--optix", "--quality-profile", "balanced",
        "--num-images", "1000", "--num-workers", "4",
        "--worker-id", "$worker", "--seed", "42"
    )
}
$jobs | Wait-Process
python .\merge.py
python .\draw_bbox.py --limit 10
python .\validate_dataset.py
```

## Texture sistemi

Texture havuzu Blender gerektirmez:

```bash
python gen_textures.py --pool-size 200 --size 512 --seed 42 --regen
```

Her havuz girdisi şu metadata'yı taşır:

- filament/materyal profili
- ana ve ikincil renk
- desen türü
- roughness ve metallic
- texture seed'i
- albedo ve normal map yolu

`TEXTURE_DIVERSITY_WINDOW=8`, son sekiz seçimin aynı texture'ı yeniden kullanmasını engeller. Düz renkli küpler bu havuzdan bağımsızdır ve `SOLID_COLOR_PROBABILITY` ile yönetilir.

## Annotation ve metadata

YOLO satırı:

```text
0 center_x center_y width height
```

Tüm koordinatlar 640×640 çıktı tuvaline göre normalize edilir. Düşük çözünürlük sistemi önce görüntüyü küçültür, sonra aynı 640×640 tuvale büyütür; koordinat uzayı değişmez.

Cycles'ta instance attribute map içindeki `cube_*` adları küp index'ine çevrilir. `env_*` nesneleri, zemin ve diğer `category_id=0` varlıklar COCO/YOLO'ya yazılmaz. Lens distortion aktifse RGB görüntü ile instance segmentation maskesi aynı dönüşümden geçirilir.

Metadata içinde en az şu alanlar bulunur:

- frame/worker seed'i ve render kalite profili
- Arka plan kaynak türü, EXR/JPEG dosya adı, dönüş, strength ve renk sıcaklığı
- JPG/PNG için kaynak çözünürlüğü, center-crop mapping ve interpolation bilgisi
- ışık türü, konumu, enerji, renk sıcaklığı ve softness
- zemin türü, rengi, roughness, metallic ve reflection
- kamera konumu, yüksekliği, distance, elevation, azimuth, roll, focal ve DoF
- çevre nesnesi türü, boyutu, konumu ve `annotated: false`
- küp X/Y/Z dönüşleri (radyan ve derece), temas yüksekliği, materyal/desen/renk
- düşük çözünürlük boyutu, büyütme yöntemi ve JPEG kalitesi
- brightness, contrast, gamma, exposure, blur, distortion ve noise değerleri

## Test ve doğrulama

Blender gerektirmeyen kontroller:

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

BlenderProc entegrasyon testi:

```bash
blenderproc run generate.py -- --test --quality-profile fast --seed 42
python draw_bbox.py --limit 10
python validate_dataset.py
```

`validate_dataset.py` şunları denetler:

- YOLO class ve normalize bbox aralıkları
- tüm çıktıların 640×640 olması
- düşük çözünürlük sonrasında tuvalin korunması
- çevre metadata'sının `annotated: false` olması
- küplerde X/Y/Z dönüş metadata'sı ve zemin teması
- COCO'da yalnızca `category_id=1` bulunması
- her karede yalnızca bir arka plan türü bulunması ve PNG/JPG karelerinde `hdri=null` olması

## Sürüm uyumluluğu ve güvenli fallback'ler

- Blender 4.x için `BLENDER_EEVEE_NEXT`, eski sürümler için `BLENDER_EEVEE` denenir.
- AgX desteklenmiyorsa Filmic, o da yoksa Standard kullanılır.
- OptiX denoiser yoksa OpenImageDenoise denenir.
- Blender/Cycles özelliği mevcut değilse ilgili özellik güvenli biçimde atlanır.
- Lens distortion için SciPy erişilemiyorsa efekt atlanır; render ve annotation üretimi devam eder.
- Hatalı kareler `output/errors.log` dosyasına yazılır ve üretim sonraki kareyle devam eder.

## Bilinen pratik notlar

- Eevee görünürlük hesabı, çevre örtücülerini Cycles instance maskesi kadar kesin ölçemez; final veri için Cycles önerilir.
- `realistic` profili, `balanced` profile göre belirgin biçimde daha yavaştır.
- Sentetik veri gerçek robot görüntüleriyle fine-tune edilmelidir.
- Paralel mod tamamlandıktan sonra mutlaka `python merge.py` çalıştırılmalıdır.
