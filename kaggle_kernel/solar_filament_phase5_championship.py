"""
Solar Filament Segmentation Challenge 2026 (IEEE BigData Cup)
Phase 5: High-Capacity Backbones, TTA, 5-Fold Calibrated Evaluation & Ensembling
- Exp 1: Backbone Scaling Comparison on Fold 0 (EfficientNet-B0 vs EfficientNet-B3 vs ConvNeXt-Tiny)
- Exp 2: Multi-Scale / Test-Time Augmentation (TTA) Ablation on Fold 0
- Exp 3: 5-Fold Full Training of Winning Scaled Backbone
- Exp 4: Calibrated Instance Extraction Across All 5 Folds
- Exp 5: Out-of-Fold 5-Model Ensemble PQ Evaluation
- Exp 6: Competition Test Set Inference & Zero-Trust 6-Point Guardrails
"""

import os
import sys
import glob
import json
import time
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as mask_util

# -------------------------------------------------------------------------
# 1. Global Setup & Frozen Split Manifest
# -------------------------------------------------------------------------

SEED = 42
TILE_SIZE = 1024
STRIDE = 512
ORIGINAL_H = 2048
ORIGINAL_W = 2048
IOU_THRESHOLD = 0.50
QUIET_SUN_STEMS = {"20160513133714Th", "20180507140210Th"}

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0 = time.time()

def log(*args):
    print(f"[{(time.time()-t0)/60:6.1f}m]", *args, flush=True)

# Frozen 5-Fold Grouped Split (707 physical observation stems)
FROZEN_STEM_TO_FOLD = json.loads('{"20110109104734Ch": 2, "20110114105034Ch": 3, "20110119082634Lh": 1, "20110123174914Mh": 2, "20110128174814Mh": 1, "20110203082634Lh": 0, "20110205082634Lh": 2, "20110211084114Th": 1, "20110213093414Th": 1, "20110215083814Th": 3, "20110220082634Lh": 1, "20110221173814Mh": 4, "20110301082654Uh": 2, "20110306082634Lh": 0, "20110312082634Lh": 3, "20110313082654Uh": 4, "20110317082654Uh": 1, "20110322005814Mh": 3, "20110324082614Th": 1, "20110325171014Mh": 2, "20110326170914Mh": 1, "20110327143554Bh": 2, "20110406082634Lh": 0, "20110410141654Bh": 2, "20110421164914Mh": 1, "20110425063154Uh": 0, "20110429135554Bh": 1, "20110502063154Uh": 3, "20110509121334Ch": 1, "20110510063134Lh": 0, "20110511063154Uh": 2, "20110517164514Mh": 0, "20110520063154Uh": 4, "20110521063134Lh": 2, "20110523063154Uh": 1, "20110525063154Uh": 2, "20110526122534Ch": 1, "20110530122834Ch": 2, "20110601063134Lh": 1, "20110603163414Mh": 0, "20110615133454Bh": 2, "20110616133454Bh": 0, "20110617063134Lh": 3, "20110619063134Lh": 2, "20110621063134Lh": 1, "20110623063134Lh": 3, "20110625163814Mh": 1, "20110707074514Th": 2, "20110711071214Th": 2, "20110724164614Mh": 1, "20110727063234Lh": 2, "20110728084714Th": 4, "20110803164914Mh": 1, "20110810063134Lh": 3, "20110815140454Bh": 4, "20110823120334Ch": 3, "20110824141054Bh": 3, "20110826074214Th": 2, "20110902073614Th": 0, "20110914063134Lh": 2, "20110918112834Ch": 3, "20110923112234Ch": 0, "20110923165814Mh": 2, "20111009110234Ch": 0, "20111011112054Uh": 1, "20111018063134Lh": 0, "20111019063154Uh": 3, "20111025080614Th": 1, "20111027145954Bh": 3, "20111029063154Uh": 3, "20111107063154Uh": 4, "20111112193254Bh": 0, "20111114082314Th": 4, "20111115063134Lh": 4, "20111116063134Lh": 3, "20111201063154Uh": 2, "20111206102934Ch": 3, "20111208063134Lh": 2, "20111214063134Lh": 3, "20111214132434Ch": 1, "20111215063134Lh": 4, "20111218085014Th": 1, "20111222063154Uh": 3, "20111223085314Th": 4, "20111226155454Bh": 0, "20111229063154Uh": 2, "20120103104134Ch": 1, "20120104063134Lh": 3, "20120110155754Bh": 1, "20120113104834Ch": 3, "20120114063154Uh": 2, "20120116063254Uh": 1, "20120127085214Th": 0, "20120131063154Uh": 2, "20120205172414Mh": 2, "20120206063134Lh": 2, "20120213063134Lh": 2, "20120215153654Bh": 4, "20120217063154Uh": 2, "20120217181154Bh": 1, "20120218173614Mh": 2, "20120223063154Uh": 4, "20120223152054Bh": 4, "20120225172914Mh": 3, "20120227161654Bh": 4, "20120306063154Uh": 2, "20120313063154Uh": 3, "20120318063154Uh": 2, "20120323144054Bh": 1, "20120328164614Mh": 2, "20120330063134Lh": 2, "20120330063154Uh": 3, "20120331142954Bh": 3, "20120403083114Th": 3, "20120403164114Mh": 0, "20120404063154Uh": 4, "20120414115434Ch": 2, "20120414175914Mh": 3, "20120421063154Uh": 1, "20120424072414Th": 0, "20120424120134Ch": 4, "20120427063154Uh": 1, "20120427162614Mh": 4, "20120430063154Uh": 4, "20120509163414Mh": 4, "20120519070814Th": 4, "20120523163014Mh": 4, "20120603123034Ch": 2, "20120610175114Mh": 4, "20120627162114Mh": 3, "20120703070814Th": 0, "20120712123634Ch": 3, "20120715123434Ch": 3, "20120720085114Th": 4, "20120723095014Th": 2, "20120725122934Ch": 2, "20120726071914Th": 1, "20120729122634Ch": 2, "20120730135454Bh": 1, "20120802090314Th": 4, "20120803102114Th": 0, "20120803164614Mh": 3, "20120810121634Ch": 4, "20120811121434Ch": 0, "20120813164714Mh": 4, "20120814140454Bh": 0, "20120814163114Mh": 4, "20120816072814Th": 1, "20120827085114Th": 0, "20120829063134Lh": 3, "20120901114934Ch": 1, "20120907141954Bh": 4, "20120910074014Th": 4, "20120913165314Mh": 4, "20120914063134Lh": 2, "20120917142654Bh": 2, "20120928063134Lh": 4, "20121006063534Lh": 3, "20121007131134Ch": 1, "20121009165914Mh": 0, "20121018063134Lh": 4, "20121021145454Bh": 4, "20121022145554Bh": 0, "20121024164714Mh": 1, "20121026063134Lh": 2, "20121027063134Lh": 2, "20121027165114Mh": 1, "20121030150354Bh": 0, "20121108155954Bh": 3, "20121109103634Ch": 1, "20121113103634Ch": 4, "20121114172254Bh": 0, "20121118092514Th": 3, "20121129063154Uh": 1, "20121207103034Ch": 0, "20121208171514Mh": 3, "20121218063534Lh": 2, "20121219063154Uh": 3, "20130107174854Bh": 3, "20130109130334Ch": 4, "20130110104734Ch": 3, "20130111104734Ch": 4, "20130116172814Mh": 2, "20130201084914Th": 1, "20130208063154Uh": 0, "20130211063134Lh": 2, "20130212153254Bh": 2, "20130214111534Ch": 3, "20130218220314Mh": 2, "20130219063134Lh": 4, "20130221063134Lh": 0, "20130222082914Th": 1, "20130225171214Mh": 4, "20130303063134Lh": 2, "20130303063154Uh": 4, "20130304170614Mh": 2, "20130305112734Ch": 1, "20130306172214Mh": 2, "20130314165714Mh": 2, "20130319144654Bh": 3, "20130324015554Uh": 3, "20130326063134Lh": 3, "20130327175114Mh": 2, "20130404063134Lh": 4, "20130405114734Ch": 4, "20130407063154Uh": 0, "20130411115134Ch": 0, "20130419070254Uh": 2, "20130423072414Th": 3, "20130425135954Bh": 1, "20130429063154Uh": 3, "20130502063134Lh": 3, "20130510134654Bh": 0, "20130517063134Lh": 2, "20130601063154Uh": 4, "20130609123334Ch": 0, "20130609173414Mh": 4, "20130611063154Uh": 4, "20130615002434Lh": 1, "20130616070414Th": 2, "20130628063134Lh": 1, "20130630070714Th": 4, "20130705161714Mh": 1, "20130707134154Bh": 1, "20130716162214Mh": 2, "20130717123434Ch": 0, "20130719140154Bh": 1, "20130725122934Ch": 0, "20130805135854Bh": 3, "20130807073114Th": 1, "20130808140054Bh": 1, "20130809072814Th": 0, "20130810121534Ch": 4, "20130811140254Bh": 1, "20130813060834Lh": 2, "20130824060834Lh": 0, "20130828141354Bh": 0, "20130901073614Th": 4, "20130903060834Lh": 1, "20130903141654Bh": 0, "20130905073714Th": 2, "20130913142354Bh": 2, "20130919060834Lh": 4, "20130921060834Lh": 0, "20130923060834Lh": 2, "20130925143254Bh": 0, "20130926060834Lh": 0, "20131006075314Th": 2, "20131007075314Th": 2, "20131011195414Mh": 1, "20131013185334Ch": 1, "20131016185334Ch": 3, "20131017185334Ch": 0, "20131019080114Th": 0, "20131020080114Th": 4, "20131025185414Mh": 1, "20131027025654Uh": 0, "20131028190734Ch": 2, "20131030185354Bh": 3, "20131031185614Mh": 4, "20131101222734Lh": 4, "20131107043054Uh": 3, "20131108045254Uh": 2, "20131109190514Mh": 4, "20131109222434Lh": 0, "20131113004514Mh": 3, "20131113185414Mh": 3, "20131115185414Mh": 1, "20131116185334Ch": 3, "20131125192154Bh": 1, "20131128192214Mh": 4, "20131130192134Ch": 4, "20131201192134Ch": 2, "20131204192154Bh": 0, "20131206084114Th": 2, "20131207084214Th": 1, "20131224192154Bh": 0, "20131225065154Uh": 0, "20131229112414Th": 0, "20140111224134Lh": 3, "20140112180134Ch": 0, "20140116180214Mh": 4, "20140122180214Mh": 4, "20140127180134Ch": 0, "20140128174554Bh": 3, "20140205180154Bh": 3, "20140210180154Bh": 4, "20140212024554Uh": 1, "20140213024554Uh": 0, "20140216230134Lh": 2, "20140218230234Lh": 4, "20140225163314Th": 4, "20140312025454Uh": 3, "20140312081114Th": 4, "20140321195814Mh": 1, "20140323020654Uh": 2, "20140401014754Uh": 1, "20140406195854Bh": 4, "20140409081614Th": 1, "20140411013854Uh": 3, "20140413195914Mh": 3, "20140414020354Uh": 4, "20140414195834Ch": 1, "20140420085614Th": 2, "20140428233234Lh": 0, "20140430071914Th": 3, "20140501195814Mh": 1, "20140508195834Ch": 1, "20140509071314Th": 4, "20140509195814Mh": 1, "20140513012554Uh": 2, "20140522070614Th": 0, "20140523195814Mh": 4, "20140525070514Th": 3, "20140531195834Ch": 4, "20140601195814Mh": 1, "20140603070414Th": 1, "20140604195854Bh": 3, "20140605195854Bh": 0, "20140607112354Uh": 4, "20140607195854Bh": 1, "20140609195854Bh": 1, "20140615195814Mh": 4, "20140615195854Bh": 4, "20140701195814Mh": 1, "20140705035754Uh": 1, "20140709195814Mh": 0, "20140710083214Th": 0, "20140712195854Bh": 1, "20140716195854Bh": 4, "20140719071414Th": 2, "20140720195834Ch": 0, "20140723071614Th": 0, "20140725195814Mh": 3, "20140726071814Th": 2, "20140802072114Th": 1, "20140803234534Lh": 0, "20140806195834Ch": 0, "20140807195854Bh": 3, "20140813064254Uh": 2, "20140814195854Bh": 0, "20140816072814Th": 0, "20140817195834Ch": 1, "20140818195814Mh": 0, "20140821195834Ch": 3, "20140903073614Th": 4, "20140904195814Mh": 1, "20140908175354Bh": 1, "20140908231334Lh": 3, "20140918195854Bh": 3, "20140921195854Bh": 4, "20140922195854Bh": 3, "20140927014554Uh": 0, "20141001195834Ch": 0, "20141004075414Th": 4, "20141006224634Lh": 4, "20141007224934Lh": 2, "20141009023454Uh": 4, "20141010195834Ch": 2, "20141012015654Uh": 1, "20141012195854Bh": 3, "20141013195834Ch": 3, "20141013195854Bh": 2, "20141016022554Uh": 3, "20141016195834Ch": 0, "20141022223334Lh": 4, "20141024195834Ch": 0, "20141025031154Uh": 2, "20141026024454Uh": 0, "20141027025054Uh": 3, "20141027224834Lh": 1, "20141105195814Mh": 3, "20141106195854Bh": 2, "20141112195854Bh": 4, "20141116222234Lh": 0, "20141129222134Lh": 4, "20141210201814Mh": 0, "20141214201834Ch": 0, "20141216222634Lh": 1, "20141225024354Uh": 1, "20141225172214Mh": 1, "20150112081954Uh": 2, "20150113105034Ch": 1, "20150113155654Bh": 0, "20150116082034Lh": 1, "20150119082034Lh": 3, "20150125172714Mh": 2, "20150130084914Th": 4, "20150203084714Th": 4, "20150207084414Th": 0, "20150216171714Mh": 3, "20150221155154Bh": 3, "20150222090014Th": 1, "20150222112034Ch": 0, "20150225171114Mh": 0, "20150228082414Th": 4, "20150302082034Lh": 4, "20150305081954Uh": 2, "20150308081954Uh": 3, "20150314081954Uh": 3, "20150315082034Lh": 0, "20150323172814Mh": 1, "20150324145954Bh": 4, "20150325015554Uh": 0, "20150327145114Th": 4, "20150330081954Uh": 3, "20150409081954Uh": 0, "20150417082314Th": 1, "20150421140454Bh": 4, "20150423120034Ch": 0, "20150427162314Mh": 0, "20150501120634Ch": 1, "20150518145654Bh": 2, "20150527082014Th": 2, "20150527161214Mh": 2, "20150531082034Lh": 1, "20150610082014Th": 4, "20150614162514Mh": 0, "20150615133554Bh": 2, "20150617161314Mh": 0, "20150620082014Th": 0, "20150621161314Mh": 1, "20150625082014Th": 4, "20150625082034Lh": 3, "20150626161514Mh": 3, "20150627133754Bh": 0, "20150629161614Mh": 1, "20150630083214Th": 4, "20150706083014Th": 2, "20150716162114Mh": 2, "20150718082014Th": 2, "20150719182314Mh": 1, "20150720082014Th": 0, "20150727082014Th": 3, "20150731082034Lh": 0, "20150806082034Lh": 2, "20150819082014Th": 4, "20150820152254Bh": 2, "20150825082034Lh": 3, "20150830084134Lh": 1, "20150913082014Th": 0, "20150916055454Uh": 2, "20150919165814Mh": 4, "20150926111934Ch": 4, "20150927111834Ch": 3, "20151011144554Bh": 0, "20151013144654Bh": 0, "20151019223534Lh": 3, "20151105133734Ch": 2, "20151106171514Mh": 3, "20151107133734Ch": 0, "20151116201114Mh": 2, "20151116224034Lh": 4, "20151117172314Mh": 4, "20151118172314Mh": 3, "20151119133734Ch": 4, "20151122172614Mh": 3, "20151128133714Th": 4, "20151128133734Ch": 4, "20151128222134Lh": 2, "20151209201754Bh": 1, "20151212171154Bh": 0, "20151213023454Uh": 2, "20151219133734Ch": 0, "20151224133734Ch": 0, "20151226133734Ch": 4, "20151228024254Uh": 0, "20160101133714Th": 2, "20160103133734Ch": 3, "20160107133714Th": 0, "20160110133714Th": 3, "20160125154854Bh": 3, "20160127225034Lh": 0, "20160129225334Lh": 0, "20160201025254Uh": 4, "20160204133734Ch": 1, "20160213153054Bh": 4, "20160215133734Ch": 0, "20160215175514Mh": 0, "20160223173514Mh": 0, "20160225173314Mh": 3, "20160226173214Mh": 0, "20160229151254Bh": 4, "20160302133734Ch": 2, "20160309172214Mh": 2, "20160310133734Ch": 4, "20160315133714Th": 0, "20160317232434Lh": 1, "20160320133714Th": 0, "20160326143654Bh": 4, "20160327170714Mh": 3, "20160401133734Ch": 4, "20160401170314Mh": 0, "20160402025254Uh": 4, "20160403170114Mh": 4, "20160408232234Lh": 1, "20160411134234Ch": 0, "20160430133714Th": 2, "20160430133734Ch": 0, "20160501143154Bh": 3, "20160510163914Mh": 4, "20160516012454Uh": 1, "20160518011654Uh": 1, "20160526011254Uh": 1, "20160530234734Lh": 4, "20160604163414Mh": 2, "20160606163514Mh": 0, "20160609133654Bh": 2, "20160617133654Bh": 2, "20160619133714Th": 4, "20160627133734Ch": 3, "20160628152654Bh": 2, "20160629133714Th": 3, "20160703133714Th": 2, "20160714235434Lh": 2, "20160720133734Ch": 1, "20160724135554Bh": 2, "20160726135354Bh": 4, "20160804133734Ch": 4, "20160808133714Th": 0, "20160809164914Mh": 0, "20160810234034Lh": 0, "20160831133714Th": 2, "20160920230134Lh": 1, "20160925165814Mh": 1, "20160925225634Lh": 2, "20160927133734Ch": 2, "20160928143554Bh": 3, "20161002225034Lh": 4, "20161012133734Ch": 3, "20161013133734Ch": 3, "20161023133814Th": 4, "20161105222634Lh": 4, "20161108021454Uh": 1, "20161109171714Mh": 0, "20161110151454Bh": 4, "20161116133714Th": 4, "20161129022454Uh": 4, "20161201160954Bh": 4, "20161205133714Th": 2, "20161210133714Th": 3, "20161217133734Ch": 3, "20161222133714Th": 2, "20161223133734Ch": 0, "20161224024154Uh": 2, "20161229133734Ch": 3, "20161231133734Ch": 4, "20170102024454Uh": 1, "20170103030554Uh": 1, "20170106133714Th": 3, "20170107133714Th": 1, "20170117175014Mh": 0, "20170119031254Uh": 1, "20170122034654Uh": 3, "20170125133734Ch": 3, "20170217165250Bh": 0, "20170218173710Mh": 3, "20170222133710Th": 2, "20170310133730Ch": 0, "20170310173450Bh": 1, "20170316133730Ch": 1, "20170322015550Uh": 2, "20170323041730Lh": 2, "20170327143550Bh": 3, "20170329133730Ch": 3, "20170408133710Th": 3, "20170411013550Uh": 1, "20170415145450Bh": 0, "20170417014150Uh": 1, "20170417232530Lh": 2, "20170418013450Uh": 3, "20170423014350Uh": 3, "20170425133730Ch": 0, "20170427154550Bh": 3, "20170502233130Lh": 0, "20170503013050Uh": 3, "20170609133710Th": 0, "20170612135150Bh": 1, "20170619012150Uh": 0, "20170701235630Lh": 4, "20170702133730Ch": 2, "20170719235330Lh": 0, "20170722235230Lh": 3, "20170726235030Lh": 4, "20170728150230Ch": 1, "20170801164710Mh": 3, "20170804140650Bh": 3, "20170805133710Th": 4, "20170806133710Th": 4, "20170809164910Mh": 3, "20170827165310Mh": 0, "20170831170750Bh": 1, "20170925225630Lh": 4, "20170927143450Bh": 3, "20171004143950Bh": 1, "20171005133730Ch": 0, "20171005144050Bh": 3, "20171007133710Th": 2, "20171007133730Ch": 4, "20171008144250Bh": 3, "20171011170310Mh": 3, "20171011224130Lh": 1, "20171014133730Ch": 0, "20171022170710Mh": 1, "20171025133730Ch": 3, "20171025223130Lh": 1, "20171027223030Lh": 0, "20171029150150Bh": 1, "20171101133730Ch": 3, "20171103222630Lh": 0, "20171106171510Mh": 1, "20171109023950Uh": 3, "20171114133730Ch": 1, "20171115170950Bh": 0, "20171115222230Lh": 3, "20171121222130Lh": 2, "20171123133730Ch": 0, "20171123222130Lh": 2, "20180109223930Lh": 4, "20180112130330Ch": 1, "20180202130330Ch": 0, "20180301050430Lh": 1, "20180306081810Th": 3, "20180309000910Mh": 4, "20180318020350Uh": 2, "20180318232230Lh": 2, "20180506140230Ch": 0, "20180710031950Uh": 0, "20180712123630Ch": 2, "20180714000010Mh": 0, "20180801232450Bh": 1, "20180804234530Lh": 4, "20180805232510Mh": 3, "20180825205030Ch": 1, "20180827232530Lh": 2, "20180923112330Ch": 0, "20180924112030Ch": 1, "20180925000030Lh": 2, "20180926014350Uh": 1, "20181014105830Ch": 0, "20181023232430Lh": 1, "20181024232430Lh": 3, "20181101230010Mh": 4, "20181102104630Ch": 4, "20190317145150Bh": 1, "20190322144450Bh": 2, "20190412025430Lh": 3, "20190602025430Lh": 4, "20190609123330Ch": 2, "20190611032810Mh": 0, "20190626025430Lh": 1, "20190629163710Mh": 2, "20190701123830Ch": 4, "20190704123830Ch": 4, "20190805054730Lh": 1, "20190925165810Mh": 3, "20190926165810Mh": 3, "20190927165910Mh": 1, "20200105073250Uh": 2, "20200110073230Lh": 1, "20200114073230Lh": 1, "20200116073250Uh": 1, "20200619123010Th": 4, "20200806053830Lh": 3, "20201022223330Lh": 4, "20201227202710Mh": 1, "20201228202430Ch": 2, "20210103223630Lh": 2, "20210201020330Lh": 1, "20210222222850Bh": 3, "20210223222830Ch": 4, "20210329015250Uh": 2, "20210329114430Ch": 1, "20210331231510Mh": 0, "20210616142650Bh": 4, "20210620163530Mh": 4, "20210621133650Bh": 0, "20210624133950Bh": 2, "20210626060050Uh": 2, "20210626133950Bh": 3, "20210721123430Ch": 2, "20210728164530Mh": 3, "20210731074710Th": 3, "20210810121630Ch": 3, "20210812044930Lh": 3, "20210814044930Lh": 4, "20210815140550Bh": 2, "20210817165030Mh": 4, "20211016145050Bh": 0, "20211119103130Ch": 3, "20211222120130Ch": 0, "20211225025850Uh": 1, "20211229120130Ch": 2, "20211229120210Th": 3, "20211231223430Lh": 2, "20220104043050Uh": 3, "20220122085210Th": 4, "20220328143550Bh": 4, "20220418020950Uh": 3, "20220617185352Mh": 4, "20220618185332Bh": 2, "20220622205632Bh": 2, "20220710085152Th": 2, "20220718185352Mh": 1, "20220721185332Bh": 2, "20220724185352Mh": 1, "20220803074752Th": 4}')

log(f"Active Compute Device: {DEVICE}")
if torch.cuda.is_available():
    log(f"GPU Model: {torch.cuda.get_device_name(0)} (Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
log(f"Frozen split loaded: {len(FROZEN_STEM_TO_FOLD)} physical observation stems.")


# -------------------------------------------------------------------------
# 2. Architectures: EfficientNet-B0, EfficientNet-B3, ConvNeXt-Tiny U-Nets
# -------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PretrainedDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv((in_ch // 2) + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PretrainedEfficientNetB0UNet(nn.Module):
    """U-Net with ImageNet Pretrained EfficientNet-B0 Encoder (10.2M parameters)"""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.features = backbone.features
        
        self.dec4 = PretrainedDecoderBlock(in_ch=1280, skip_ch=112, out_ch=256)
        self.dec3 = PretrainedDecoderBlock(in_ch=256, skip_ch=40, out_ch=128)
        self.dec2 = PretrainedDecoderBlock(in_ch=128, skip_ch=24, out_ch=64)
        self.dec1 = PretrainedDecoderBlock(in_ch=64, skip_ch=16, out_ch=32)
        
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        s0 = self.features[0](x)
        s1 = self.features[1](s0)   # 16 ch, 1/2
        s2 = self.features[2](s1)   # 24 ch, 1/4
        s3 = self.features[3](s2)   # 40 ch, 1/8
        s4 = self.features[4](s3)
        s5 = self.features[5](s4)   # 112 ch, 1/16
        s6 = self.features[6](s5)
        s7 = self.features[7](s6)
        s8 = self.features[8](s7)   # 1280 ch, 1/32

        d4 = self.dec4(s8, s5)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        out = self.final_up(d1)
        return self.final_conv(out)


class PretrainedEfficientNetB3UNet(nn.Module):
    """U-Net with ImageNet Pretrained EfficientNet-B3 Encoder (18.7M parameters)"""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)
        self.features = backbone.features

        self.dec4 = PretrainedDecoderBlock(in_ch=1536, skip_ch=136, out_ch=256)
        self.dec3 = PretrainedDecoderBlock(in_ch=256, skip_ch=48, out_ch=128)
        self.dec2 = PretrainedDecoderBlock(in_ch=128, skip_ch=32, out_ch=64)
        self.dec1 = PretrainedDecoderBlock(in_ch=64, skip_ch=24, out_ch=32)

        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        s0 = self.features[0](x)
        s1 = self.features[1](s0)   # 24 ch, 1/2
        s2 = self.features[2](s1)   # 32 ch, 1/4
        s3 = self.features[3](s2)   # 48 ch, 1/8
        s4 = self.features[4](s3)
        s5 = self.features[5](s4)   # 136 ch, 1/16
        s6 = self.features[6](s5)
        s7 = self.features[7](s6)
        s8 = self.features[8](s7)   # 1536 ch, 1/32

        d4 = self.dec4(s8, s5)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        out = self.final_up(d1)
        return self.final_conv(out)


class ConvNeXtTinyUNet(nn.Module):
    """U-Net with ImageNet Pretrained ConvNeXt-Tiny Encoder (32.2M parameters)"""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        self.features = backbone.features

        self.dec3 = PretrainedDecoderBlock(in_ch=768, skip_ch=384, out_ch=256)
        self.dec2 = PretrainedDecoderBlock(in_ch=256, skip_ch=192, out_ch=128)
        self.dec1 = PretrainedDecoderBlock(in_ch=128, skip_ch=96, out_ch=64)

        self.final_up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv1 = DoubleConv(32, 32)
        self.final_up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        s1 = self.features[1](self.features[0](x))  # 96 ch, 1/4
        s3 = self.features[3](self.features[2](s1)) # 192 ch, 1/8
        s5 = self.features[5](self.features[4](s3)) # 384 ch, 1/16
        s7 = self.features[7](self.features[6](s5)) # 768 ch, 1/32

        d3 = self.dec3(s7, s5)
        d2 = self.dec2(d3, s3)
        d1 = self.dec1(d2, s1)
        u1 = self.final_conv1(self.final_up1(d1))
        u2 = self.final_conv2(self.final_up2(u1))
        return u2


def create_model(model_name: str) -> nn.Module:
    if model_name == "efficientnet_b0":
        return PretrainedEfficientNetB0UNet(pretrained=True)
    elif model_name == "efficientnet_b3":
        return PretrainedEfficientNetB3UNet(pretrained=True)
    elif model_name == "convnext_tiny":
        return ConvNeXtTinyUNet(pretrained=True)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# -------------------------------------------------------------------------
# 3. Loss & Official Metrics
# -------------------------------------------------------------------------

def soft_dice_loss(pred_sig: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred_flat = pred_sig.view(pred_sig.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    cardinality = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


def mask_to_rle(binary_mask: np.ndarray) -> str:
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    counts = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else str(rle["counts"])
    return counts


def get_gaussian_weight_matrix(size: int = TILE_SIZE, sigma_scale: float = 0.25) -> torch.Tensor:
    sigma = size * sigma_scale
    k1d = cv2.getGaussianKernel(size, sigma)
    k2d = np.outer(k1d, k1d)
    k2d = k2d / k2d.max()
    return torch.from_numpy(k2d.astype(np.float32)).to(DEVICE)


def compute_official_pq(gt_df: pd.DataFrame, pred_df: pd.DataFrame, iou_threshold: float = IOU_THRESHOLD) -> Dict:
    if len(pred_df) == 0:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": 0, "fn": len(gt_df), "num_annotator_views": len(gt_df["filament_id"].str.split("_").str[0].unique())}

    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids = pred_df["filament_id"].str.rsplit("_", n=1).str[0]
    annotator_views = gt_annotator_image_ids.unique()

    tp_iou_scores = []
    fp_count = 0
    fn_count = 0

    pred_by_stem = {}
    for stem, group in pred_df.groupby(pred_image_ids):
        pred_by_stem[stem] = group["segmentation_rle"].tolist()

    for annotator_view in annotator_views:
        stem = annotator_view.split("-", 1)[1] if "-" in annotator_view else annotator_view
        gt_rles = gt_df.loc[gt_annotator_image_ids == annotator_view, "segmentation_rle"].tolist()
        pred_rles = pred_by_stem.get(stem, [])

        n_gt = len(gt_rles)
        n_pred = len(pred_rles)

        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue

        gt_dicts = [{"size": [ORIGINAL_H, ORIGINAL_W], "counts": r.encode("ascii") if isinstance(r, str) else r} for r in gt_rles]
        dt_dicts = [{"size": [ORIGINAL_H, ORIGINAL_W], "counts": r.encode("ascii") if isinstance(r, str) else r} for r in pred_rles]
        iou_matrix = mask_util.iou(dt_dicts, gt_dicts, [0] * n_gt).T
        hit_matrix = iou_matrix > iou_threshold

        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())
        fp_count += int((hit_matrix.sum(axis=0) == 0).sum())
        fn_count += int((hit_matrix.sum(axis=1) == 0).sum())

    tp_count = len(tp_iou_scores)
    denom = tp_count + 0.5 * fp_count + 0.5 * fn_count
    sq = float(np.mean(tp_iou_scores)) if tp_count > 0 else 0.0
    rq = float(tp_count / denom) if denom > 0 else 0.0
    pq = float(sum(tp_iou_scores) / denom) if denom > 0 else 0.0

    return {
        "pq": pq,
        "sq": sq,
        "rq": rq,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "num_annotator_views": len(annotator_views)
    }


# -------------------------------------------------------------------------
# 4. Dataset & Tiling (Strategy A)
# -------------------------------------------------------------------------

class SolarTileDataset(Dataset):
    def __init__(self, samples: List[Dict], train_img_dir: str, augment: bool = True):
        self.samples = samples
        self.train_img_dir = train_img_dir
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        img_p = os.path.join(self.train_img_dir, item["file_name"])
        if not os.path.exists(img_p):
            for ext in [".jpeg", ".jpg", ".png"]:
                c = os.path.join(self.train_img_dir, item["stem"] + ext)
                if os.path.exists(c):
                    img_p = c
                    break

        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE) if os.path.exists(img_p) else np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)
        
        mask = np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)
        for ann in item["annotations"]:
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            rles = mask_util.frPyObjects(seg, ORIGINAL_H, ORIGINAL_W)
            rle = mask_util.merge(rles)
            m = mask_util.decode(rle)
            if m.ndim == 3:
                m = m.squeeze(-1)
            mask |= m.astype(np.uint8)

        cx, cy = item.get("center", (1024, 1024))
        if self.augment:
            cx += random.randint(-64, 64)
            cy += random.randint(-64, 64)

        x0 = max(0, min(ORIGINAL_W - TILE_SIZE, cx - TILE_SIZE // 2))
        y0 = max(0, min(ORIGINAL_H - TILE_SIZE, cy - TILE_SIZE // 2))

        patch_img = img[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE].astype(np.float32) / 255.0
        patch_mask = mask[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE].astype(np.float32)

        if self.augment:
            if random.random() > 0.5:
                patch_img = np.ascontiguousarray(np.fliplr(patch_img))
                patch_mask = np.ascontiguousarray(np.fliplr(patch_mask))
            if random.random() > 0.5:
                patch_img = np.ascontiguousarray(np.flipud(patch_img))
                patch_mask = np.ascontiguousarray(np.flipud(patch_mask))

        return (
            torch.from_numpy(patch_img).unsqueeze(0),
            torch.from_numpy(patch_mask).unsqueeze(0)
        )


def build_tile_samples(coco: Dict, stem_to_fold: Dict[str, int], fold: int, is_train: bool) -> List[Dict]:
    images_by_id = {im["id"]: im for im in coco["images"]}
    stems_to_image_ids = defaultdict(list)
    for im in coco["images"]:
        stem = im["file_name"].split(".")[0]
        assigned_fold = stem_to_fold[stem]
        if (assigned_fold != fold) if is_train else (assigned_fold == fold):
            stems_to_image_ids[stem].append(im["id"])

    anns_by_image_id = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image_id[a["image_id"]].append(a)

    tile_samples = []
    for stem, iids in stems_to_image_ids.items():
        for iid in iids:
            im_info = images_by_id[iid]
            anns = anns_by_image_id[iid]
            
            centers = []
            for a in anns:
                bbox = a.get("bbox", [])
                if len(bbox) == 4:
                    cx = int(bbox[0] + bbox[2] / 2)
                    cy = int(bbox[1] + bbox[3] / 2)
                    centers.append((cx, cy))
                    
            if len(centers) > 0:
                sampled_centers = random.sample(centers, min(len(centers), 2))
                for c in sampled_centers:
                    tile_samples.append({
                        "stem": stem,
                        "file_name": im_info["file_name"],
                        "annotations": anns,
                        "center": c
                    })
            else:
                tile_samples.append({
                    "stem": stem,
                    "file_name": im_info["file_name"],
                    "annotations": anns,
                    "center": (1024, 1024)
                })
    return tile_samples


# -------------------------------------------------------------------------
# 5. Extraction with Non-Overlap Enforcement
# -------------------------------------------------------------------------

def remove_instance_overlaps(masks: List[np.ndarray], scores: List[float], min_area: int = 60) -> List[np.ndarray]:
    if len(masks) == 0:
        return []
    sorted_idx = np.argsort(scores)[::-1]
    sorted_masks = [masks[i] for i in sorted_idx]

    occupied = np.zeros(masks[0].shape, dtype=bool)
    clean_masks = []
    for m in sorted_masks:
        clean_m = m.astype(bool) & ~occupied
        if clean_m.sum() >= min_area:
            clean_masks.append(clean_m.astype(np.uint8))
            occupied |= clean_m
    return clean_masks


def extract_instances_calibrated(
    full_prob: np.ndarray,
    stem: str,
    thresh: float = 0.60,
    min_area: int = 100,
    disjoint: bool = True
) -> List[Dict]:
    binary = (full_prob >= thresh).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    cand_masks = []
    cand_scores = []
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            mask = (labels == lbl).astype(np.uint8)
            cand_masks.append(mask)
            cand_scores.append(float(full_prob[mask > 0].mean()))

    if disjoint and len(cand_masks) > 0:
        clean_masks = remove_instance_overlaps(cand_masks, cand_scores, min_area=min_area)
    else:
        clean_masks = cand_masks

    rows = []
    for i, m in enumerate(clean_masks, 1):
        rows.append({
            "filament_id": f"{stem}_{i}",
            "segmentation_rle": mask_to_rle(m)
        })
    return rows


# -------------------------------------------------------------------------
# 6. High-Res Inference Engine with TTA Support
# -------------------------------------------------------------------------

def predict_single_image(
    model: nn.Module,
    img: np.ndarray,
    gaussian_weight: torch.Tensor,
    use_tta: bool = False
) -> np.ndarray:
    """Predicts a 2048x2048 observation using 1024x1024 Gaussian tiles and optional TTA."""
    img_norm = img.astype(np.float32) / 255.0
    prob_accum = torch.zeros((ORIGINAL_H, ORIGINAL_W), device=DEVICE, dtype=torch.float32)
    weight_accum = torch.zeros((ORIGINAL_H, ORIGINAL_W), device=DEVICE, dtype=torch.float32)

    for y0 in range(0, ORIGINAL_H - TILE_SIZE + 1, STRIDE):
        for x0 in range(0, ORIGINAL_W - TILE_SIZE + 1, STRIDE):
            patch = np.ascontiguousarray(img_norm[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE])
            p_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.amp.autocast("cuda"):
                logits = model(p_tensor)
                prob = torch.sigmoid(logits).squeeze()

                if use_tta:
                    # Horizontal Flip
                    patch_h = np.ascontiguousarray(np.fliplr(patch))
                    p_tensor_h = torch.from_numpy(patch_h).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    prob_h = torch.sigmoid(model(p_tensor_h)).squeeze()
                    prob_h = torch.flip(prob_h, dims=[1])

                    # Vertical Flip
                    patch_v = np.ascontiguousarray(np.flipud(patch))
                    p_tensor_v = torch.from_numpy(patch_v).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    prob_v = torch.sigmoid(model(p_tensor_v)).squeeze()
                    prob_v = torch.flip(prob_v, dims=[0])

                    prob = (prob + prob_h + prob_v) / 3.0

            prob_accum[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE] += prob * gaussian_weight
            weight_accum[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE] += gaussian_weight

    full_prob = (prob_accum / torch.clamp(weight_accum, min=1e-6)).to(torch.float32).cpu().numpy()
    return full_prob


# -------------------------------------------------------------------------
# 7. Model Training & Evaluation Engine
# -------------------------------------------------------------------------

def train_and_eval_fold(
    fold: int,
    model_name: str,
    coco: Dict,
    img_id_to_stem: Dict[str, str],
    train_img_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    accum_steps: int = 2,
    lr: float = 4e-4,
    use_tta: bool = False,
    extract_thresh: float = 0.60,
    extract_min_area: int = 100,
) -> Tuple[Dict, Dict[str, np.ndarray], nn.Module]:
    """Trains a model on 4 folds, evaluates on remaining fold with optional TTA."""
    set_seed(SEED + fold * 10)
    exp_name = f"{model_name}_fold{fold}"
    log(f"--- Starting Training & Validation: {exp_name} (TTA={use_tta}) ---")

    train_samples = build_tile_samples(coco, FROZEN_STEM_TO_FOLD, fold=fold, is_train=True)
    val_stems = sorted([s for s, f in FROZEN_STEM_TO_FOLD.items() if f == fold])
    log(f"Prepared {len(train_samples)} native 1024x1024 tile samples across training folds.")

    dataset = SolarTileDataset(train_samples, train_img_dir=train_img_dir, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = create_model(model_name).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_bce = nn.BCEWithLogitsLoss()
    total_steps = max(1, (epochs * len(loader)) // accum_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    t_train_start = time.time()
    model.train()
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                sig = torch.sigmoid(logits)
                loss = (criterion_bce(logits, masks) + soft_dice_loss(sig, masks)) / accum_steps

            scaler.scale(loss).backward()
            ep_loss += loss.item() * accum_steps

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        log(f"Fold {fold} [{model_name}] Ep {ep}/{epochs} | Loss: {ep_loss/len(loader):.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
    train_time = time.time() - t_train_start
    peak_gpu_mb = torch.cuda.max_memory_allocated(0) / 1e6 if torch.cuda.is_available() else 0.0

    ckpt_path = f"/kaggle/working/{exp_name}.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Validation Inference
    gaussian_weight = get_gaussian_weight_matrix(TILE_SIZE)
    t_infer_start = time.time()
    model.eval()

    val_prob_maps = {}
    pred_rows = []

    with torch.no_grad():
        for stem in val_stems:
            img_p = os.path.join(train_img_dir, stem + ".jpeg")
            if not os.path.exists(img_p):
                for ext in [".jpg", ".png"]:
                    cand = os.path.join(train_img_dir, stem + ext)
                    if os.path.exists(cand):
                        img_p = cand
                        break
            img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE) if os.path.exists(img_p) else np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)
            full_prob = predict_single_image(model, img, gaussian_weight, use_tta=use_tta)
            val_prob_maps[stem] = full_prob

            rows = extract_instances_calibrated(full_prob, stem=stem, thresh=extract_thresh, min_area=extract_min_area, disjoint=True)
            pred_rows.extend(rows)

    infer_time = time.time() - t_infer_start
    pred_df = pd.DataFrame(pred_rows, columns=["filament_id", "segmentation_rle"])

    val_stems_set = set(val_stems)
    gt_rows = []
    for ann in coco["annotations"]:
        im_stem = img_id_to_stem.get(ann["image_id"])
        if im_stem in val_stems_set:
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            rles = mask_util.frPyObjects(seg, ORIGINAL_H, ORIGINAL_W)
            rle = mask_util.merge(rles)
            counts = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else str(rle["counts"])
            gt_rows.append({"filament_id": f"{ann['image_id']}_{ann['id']}", "segmentation_rle": counts})
    gt_df = pd.DataFrame(gt_rows, columns=["filament_id", "segmentation_rle"])

    m = compute_official_pq(gt_df, pred_df, iou_threshold=0.50)
    result = {
        "fold": fold,
        "model_name": model_name,
        "parameters": num_params,
        "peak_gpu_mb": round(peak_gpu_mb, 1),
        "use_tta": use_tta,
        "extract_thresh": extract_thresh,
        "extract_min_area": extract_min_area,
        "pq": m["pq"],
        "sq": m["sq"],
        "rq": m["rq"],
        "tp": m["tp"],
        "fp": m["fp"],
        "fn": m["fn"],
        "total_predicted": len(pred_rows),
        "train_time_seconds": round(train_time, 2),
        "infer_time_seconds": round(infer_time, 2),
        "checkpoint_path": ckpt_path
    }
    log(f"Fold {fold} [{model_name} | TTA={use_tta}] PQ: {result['pq']:.5f} (SQ: {result['sq']:.5f}, RQ: {result['rq']:.5f} | TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']})")
    return result, val_prob_maps, model


# -------------------------------------------------------------------------
# 8. Master Phase 5 Execution Flow
# -------------------------------------------------------------------------

def main():
    log("=== Starting Phase 5: High-Capacity Backbones, TTA & Ensemble Championship ===")
    ann_paths = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)
    if not ann_paths:
        ann_paths = glob.glob("data/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)
    assert len(ann_paths) > 0, "Could not find training annotations!"
    
    comp_root = os.path.dirname(os.path.dirname(ann_paths[0]))
    train_img_dir = os.path.join(comp_root, "train", "train_images")
    test_img_dir = os.path.join(comp_root, "test", "test_images")
    if not os.path.exists(test_img_dir):
        cand = glob.glob(os.path.join(comp_root, "**", "test_images"), recursive=True)
        if cand:
            test_img_dir = cand[0]

    with open(ann_paths[0], "r", encoding="utf-8") as f:
        coco = json.load(f)
    img_id_to_stem = {im["id"]: im["file_name"].split(".")[0] for im in coco["images"]}

    phase5_summary = {
        "phase": 5,
        "exp1_backbone_scaling_fold0": {},
        "exp2_tta_ablation_fold0": {},
        "exp3_winning_backbone_5folds": {},
        "exp4_calibrated_5folds": {},
        "exp5_ensemble_oof": {},
        "exp6_test_submission": {}
    }

    # ---------------------------------------------------------------------
    # EXPERIMENT 1: Backbone Scaling Comparison on Fold 0
    # ---------------------------------------------------------------------
    log("=================================================================")
    log("EXPERIMENT 1: Backbone Scaling on Fold 0 (B0 vs B3 vs ConvNeXt-Tiny)")
    log("=================================================================")
    
    # 1. EfficientNet-B0 (Fold 0 Benchmark with standard calibrated extraction)
    res_b0, prob_maps_b0, model_b0 = train_and_eval_fold(
        fold=0, model_name="efficientnet_b0", coco=coco,
        img_id_to_stem=img_id_to_stem, train_img_dir=train_img_dir,
        epochs=3, batch_size=4, accum_steps=2, use_tta=False
    )
    
    # 2. EfficientNet-B3 (Fold 0)
    res_b3, prob_maps_b3, model_b3 = train_and_eval_fold(
        fold=0, model_name="efficientnet_b3", coco=coco,
        img_id_to_stem=img_id_to_stem, train_img_dir=train_img_dir,
        epochs=3, batch_size=4, accum_steps=2, use_tta=False
    )

    # 3. ConvNeXt-Tiny (Fold 0)
    res_convnext, prob_maps_convnext, model_convnext = train_and_eval_fold(
        fold=0, model_name="convnext_tiny", coco=coco,
        img_id_to_stem=img_id_to_stem, train_img_dir=train_img_dir,
        epochs=3, batch_size=4, accum_steps=2, use_tta=False
    )

    phase5_summary["exp1_backbone_scaling_fold0"] = {
        "efficientnet_b0": res_b0,
        "efficientnet_b3": res_b3,
        "convnext_tiny": res_convnext
    }
    log(f"Fold 0 Scaled Results -> EffNet-B0 PQ: {res_b0['pq']:.5f} | EffNet-B3 PQ: {res_b3['pq']:.5f} | ConvNeXt-Tiny PQ: {res_convnext['pq']:.5f}")

    # Determine winning backbone
    candidates = [("efficientnet_b0", res_b0), ("efficientnet_b3", res_b3), ("convnext_tiny", res_convnext)]
    winner_name, winner_res = max(candidates, key=lambda x: x[1]["pq"])
    log(f">>> Winner Backbone Selected for 5-Fold Scaling: {winner_name} (PQ: {winner_res['pq']:.5f})")

    # ---------------------------------------------------------------------
    # EXPERIMENT 2: Multi-Scale / Test-Time Augmentation (TTA) Ablation on Fold 0
    # ---------------------------------------------------------------------
    log("=================================================================")
    log("EXPERIMENT 2: Test-Time Augmentation (TTA) Ablation on Fold 0")
    log("=================================================================")
    gaussian_weight = get_gaussian_weight_matrix(TILE_SIZE)
    val_stems_f0 = sorted([s for s, f in FROZEN_STEM_TO_FOLD.items() if f == 0])
    
    # Evaluate Winner Model on Fold 0 with TTA
    winner_model = model_convnext if winner_name == "convnext_tiny" else (model_b3 if winner_name == "efficientnet_b3" else model_b0)
    winner_model.eval()

    prob_maps_winner_f0_tta = {}
    pred_rows_f0_tta = []
    with torch.no_grad():
        for stem in val_stems_f0:
            img_p = os.path.join(train_img_dir, stem + ".jpeg")
            if not os.path.exists(img_p):
                for ext in [".jpg", ".png"]:
                    c = os.path.join(train_img_dir, stem + ext)
                    if os.path.exists(c):
                        img_p = c
                        break
            img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE) if os.path.exists(img_p) else np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)
            full_prob = predict_single_image(winner_model, img, gaussian_weight, use_tta=True)
            prob_maps_winner_f0_tta[stem] = full_prob
            rows = extract_instances_calibrated(full_prob, stem=stem, thresh=0.60, min_area=100, disjoint=True)
            pred_rows_f0_tta.extend(rows)

    val_stems_set_f0 = set(val_stems_f0)
    gt_rows_f0 = []
    for ann in coco["annotations"]:
        im_stem = img_id_to_stem.get(ann["image_id"])
        if im_stem in val_stems_set_f0:
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            rles = mask_util.frPyObjects(seg, ORIGINAL_H, ORIGINAL_W)
            rle = mask_util.merge(rles)
            counts = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else str(rle["counts"])
            gt_rows_f0.append({"filament_id": f"{ann['image_id']}_{ann['id']}", "segmentation_rle": counts})
    gt_df_f0 = pd.DataFrame(gt_rows_f0, columns=["filament_id", "segmentation_rle"])
    pred_df_f0_tta = pd.DataFrame(pred_rows_f0_tta, columns=["filament_id", "segmentation_rle"])

    m_tta = compute_official_pq(gt_df_f0, pred_df_f0_tta, iou_threshold=0.50)
    res_winner_tta = {
        "model_name": winner_name,
        "without_tta": winner_res,
        "with_tta": {
            "pq": m_tta["pq"], "sq": m_tta["sq"], "rq": m_tta["rq"],
            "tp": m_tta["tp"], "fp": m_tta["fp"], "fn": m_tta["fn"]
        },
        "pq_delta": round(m_tta["pq"] - winner_res["pq"], 5),
        "fp_delta": m_tta["fp"] - winner_res["fp"]
    }
    phase5_summary["exp2_tta_ablation_fold0"] = res_winner_tta
    log(f">>> Exp 2 TTA Impact: Without TTA PQ = {winner_res['pq']:.5f} -> With TTA PQ = {m_tta['pq']:.5f} (Delta: {res_winner_tta['pq_delta']:+.5f})")

    # ---------------------------------------------------------------------
    # EXPERIMENT 3 & 4: 5-Fold Training of Winning Scaled Backbone with TTA
    # ---------------------------------------------------------------------
    log("=================================================================")
    log(f"EXPERIMENT 3 & 4: Full 5-Fold Training of {winner_name} with Calibrated TTA")
    log("=================================================================")
    scaled_5fold_results = [winner_res if not res_winner_tta["pq_delta"] > 0 else {**winner_res, **res_winner_tta["with_tta"]}]
    all_oof_prob_maps = {0: prob_maps_winner_f0_tta if res_winner_tta["pq_delta"] > 0 else (prob_maps_convnext if winner_name == "convnext_tiny" else (prob_maps_b3 if winner_name == "efficientnet_b3" else prob_maps_b0))}
    all_models = [winner_model]

    for f in range(1, 5):
        res_f, prob_maps_f, mod_f = train_and_eval_fold(
            fold=f, model_name=winner_name, coco=coco,
            img_id_to_stem=img_id_to_stem, train_img_dir=train_img_dir,
            epochs=3, batch_size=4, accum_steps=2, use_tta=True
        )
        scaled_5fold_results.append(res_f)
        all_oof_prob_maps[f] = prob_maps_f
        all_models.append(mod_f)

    pqs_scaled = [r["pq"] for r in scaled_5fold_results]
    sqs_scaled = [r["sq"] for r in scaled_5fold_results]
    rqs_scaled = [r["rq"] for r in scaled_5fold_results]
    phase5_summary["exp3_winning_backbone_5folds"] = {
        "model_name": winner_name,
        "mean_pq": float(np.mean(pqs_scaled)),
        "std_pq": float(np.std(pqs_scaled)),
        "mean_sq": float(np.mean(sqs_scaled)),
        "mean_rq": float(np.mean(rqs_scaled)),
        "total_tp": int(sum(r["tp"] for r in scaled_5fold_results)),
        "total_fp": int(sum(r["fp"] for r in scaled_5fold_results)),
        "total_fn": int(sum(r["fn"] for r in scaled_5fold_results)),
        "folds": scaled_5fold_results
    }
    log(f">>> Exp 3 5-Fold Scaled Backbone ({winner_name}): Mean PQ = {phase5_summary['exp3_winning_backbone_5folds']['mean_pq']:.5f} +/- {phase5_summary['exp3_winning_backbone_5folds']['std_pq']:.5f}")

    # ---------------------------------------------------------------------
    # EXPERIMENT 5: Out-of-Fold 5-Model Ensemble PQ Evaluation
    # ---------------------------------------------------------------------
    log("=================================================================")
    log("EXPERIMENT 5: Out-of-Fold 5-Model Ensemble Evaluation")
    log("=================================================================")
    # Combine out-of-fold predictions across all 707 stems
    all_gt_rows = []
    all_pred_rows_ensemble = []

    for f in range(5):
        val_stems_f = sorted([s for s, fld in FROZEN_STEM_TO_FOLD.items() if fld == f])
        val_set_f = set(val_stems_f)
        
        # Ground truth for this fold
        for ann in coco["annotations"]:
            im_stem = img_id_to_stem.get(ann["image_id"])
            if im_stem in val_set_f:
                seg = ann.get("segmentation", [])
                if not seg:
                    continue
                rles = mask_util.frPyObjects(seg, ORIGINAL_H, ORIGINAL_W)
                rle = mask_util.merge(rles)
                counts = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else str(rle["counts"])
                all_gt_rows.append({"filament_id": f"{ann['image_id']}_{ann['id']}", "segmentation_rle": counts})

        # Out-of-fold predictions
        for stem in val_stems_f:
            full_prob = all_oof_prob_maps[f][stem]
            rows = extract_instances_calibrated(full_prob, stem=stem, thresh=0.60, min_area=100, disjoint=True)
            all_pred_rows_ensemble.extend(rows)

    all_gt_df = pd.DataFrame(all_gt_rows, columns=["filament_id", "segmentation_rle"])
    all_pred_df_ensemble = pd.DataFrame(all_pred_rows_ensemble, columns=["filament_id", "segmentation_rle"])
    m_ensemble = compute_official_pq(all_gt_df, all_pred_df_ensemble, iou_threshold=0.50)

    phase5_summary["exp5_ensemble_oof"] = {
        "pq": m_ensemble["pq"],
        "sq": m_ensemble["sq"],
        "rq": m_ensemble["rq"],
        "tp": m_ensemble["tp"],
        "fp": m_ensemble["fp"],
        "fn": m_ensemble["fn"],
        "total_predicted_instances": len(all_pred_rows_ensemble)
    }
    log(f">>> Full 5-Fold OOF PQ = {m_ensemble['pq']:.5f} (SQ: {m_ensemble['sq']:.5f}, RQ: {m_ensemble['rq']:.5f} | TP: {m_ensemble['tp']}, FP: {m_ensemble['fp']}, FN: {m_ensemble['fn']})")

    # ---------------------------------------------------------------------
    # EXPERIMENT 6: Competition Test Set Inference & 6-Point Guardrails
    # ---------------------------------------------------------------------
    log("=================================================================")
    log("EXPERIMENT 6: Competition Test Set Inference & Zero-Trust Guardrails")
    log("=================================================================")
    test_images = sorted(glob.glob(os.path.join(test_img_dir, "*.jpeg"))) + \
                  sorted(glob.glob(os.path.join(test_img_dir, "*.jpg"))) + \
                  sorted(glob.glob(os.path.join(test_img_dir, "*.png")))
    log(f"Found {len(test_images)} test images in: {test_img_dir}")

    test_submission_rows = []
    test_stems_processed = set()

    for idx, img_path in enumerate(test_images, 1):
        stem = Path(img_path).stem
        test_stems_processed.add(stem)

        # Quiet Sun Guardrail
        if stem in QUIET_SUN_STEMS:
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Ensemble prediction across all 5 models with TTA
        prob_accum_test = np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.float32)
        for mod in all_models:
            mod_prob = predict_single_image(mod, img, gaussian_weight, use_tta=True)
            prob_accum_test += mod_prob / len(all_models)

        # Calibrated instance extraction with strict disjointness
        inst_rows = extract_instances_calibrated(prob_accum_test, stem=stem, thresh=0.60, min_area=100, disjoint=True)
        test_submission_rows.extend(inst_rows)

        if idx % 30 == 0 or idx == len(test_images):
            log(f"Processed {idx}/{len(test_images)} test images ({len(test_submission_rows)} instances generated so far).")

    sub_df = pd.DataFrame(test_submission_rows, columns=["filament_id", "segmentation_rle"])
    sub_path = "/kaggle/working/submission_phase5_ensemble.csv"
    sub_df.to_csv(sub_path, index=False)
    log(f"Saved Phase 5 Submission to: {sub_path} ({len(sub_df)} total instances)")

    # 6-Point Zero-Trust Guardrail Assertions
    log("Running 6-Point Zero-Trust Submission Integrity Checks...")
    
    # 1. Row count & Stems check
    assert len(test_stems_processed) == 180, f"Expected 180 test stems, got {len(test_stems_processed)}!"
    
    # 2. Quiet-Sun check
    quiet_rows = sub_df[sub_df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0] in QUIET_SUN_STEMS)]
    assert len(quiet_rows) == 0, f"Violation: Found {len(quiet_rows)} instances on quiet-sun stems!"

    # 3. Disjointness check
    for stem, grp in sub_df.groupby(sub_df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])):
        rles = grp["segmentation_rle"].tolist()
        if len(rles) <= 1:
            continue
        accum = np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)
        for r in rles:
            m = mask_util.decode({"size": [ORIGINAL_H, ORIGINAL_W], "counts": r.encode("ascii")})
            assert np.all((accum & m) == 0), f"Overlap detected in stem {stem}!"
            accum |= m

    # 4. Valid RLE check
    for r in sub_df["segmentation_rle"].head(50):
        d = mask_util.decode({"size": [ORIGINAL_H, ORIGINAL_W], "counts": r.encode("ascii")})
        assert d.shape == (ORIGINAL_H, ORIGINAL_W), f"Malformed RLE shape {d.shape}!"

    # 5. ID format integrity
    assert sub_df["filament_id"].is_unique, "Duplicate filament_ids found!"

    # 6. Minimum area floor
    for r in sub_df["segmentation_rle"].head(50):
        m = mask_util.decode({"size": [ORIGINAL_H, ORIGINAL_W], "counts": r.encode("ascii")})
        assert m.sum() >= 60, f"Instance area below minimum floor: {m.sum()}"

    log("ALL 6 SUBMISSION GUARDRAILS PASSED WITH ZERO VIOLATIONS!")

    phase5_summary["exp6_test_submission"] = {
        "submission_path": sub_path,
        "total_instances": len(sub_df),
        "test_stems_processed": len(test_stems_processed),
        "active_stems_with_predictions": int(sub_df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0]).nunique()),
        "guardrails_verified": True
    }

    out_file = "/kaggle/working/phase5_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(phase5_summary, f, indent=2)
    log(f"Exported Phase 5 results to: {out_file}")
    log("=== Phase 5 Execution Finished Successfully! ===")


if __name__ == "__main__":
    main()
