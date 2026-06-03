import argparse
import html
import json
import math
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

from test_es_visual_search import TextSearchVisualizer
from vl_backends import create_backend


COMMON_CONCEPTS = [
    {
        "benchmark_type": "concept",
        "query": "street",
        "label": "street",
        "positive_patterns": [r"\bstreet\b", r"\bstreetscape\b", r"\bavenue\b", r"\broad\b", r"\bboulevard\b", r"\blane\b", r"\bstrasse\b", r"\bstraße\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "square",
        "label": "square",
        "positive_patterns": [r"\bsquare\b", r"\bplaza\b", r"\bmarket square\b", r"\bpublic square\b", r"\bplatz\b", r"\bt[ée]r\b", r"\bpiazza\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "market",
        "label": "market",
        "positive_patterns": [r"\bmarket\b", r"\bmarketplace\b", r"\bmarket square\b", r"\bbazaar\b", r"\bfair\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "waterfront",
        "label": "waterfront",
        "positive_patterns": [r"\bwaterfront\b", r"\briverside\b", r"\bseafront\b", r"\bquay\b", r"\bembankment\b", r"\bshorefront\b", r"\bharbor\b", r"\bharbour\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "harbor",
        "label": "harbor",
        "positive_patterns": [r"\bharbor\b", r"\bharbour\b", r"\bport\b", r"\bmarina\b", r"\bdock\b", r"\bquay\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "canal",
        "label": "canal",
        "positive_patterns": [r"\bcanal\b", r"\bwaterway\b", r"\bchannel\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "bridge",
        "label": "bridge",
        "positive_patterns": [r"\bbridge\b", r"\bdrawbridge\b", r"\bviaduct\b", r"\bfootbridge\b", r"\barched bridge\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "station",
        "label": "station",
        "positive_patterns": [r"\bstation\b", r"\brailway station\b", r"\btrain station\b", r"\bterminal\b", r"\bbahnhof\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "church",
        "label": "church",
        "positive_patterns": [r"\bchurch\b", r"\bkirche\b", r"\bdomkirche\b", r"\bbasilica\b", r"\babbey\b", r"\bchapel\b", r"\bcathedral\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "cathedral",
        "label": "cathedral",
        "positive_patterns": [r"\bcathedral\b", r"\bdom\b", r"\bduomo\b", r"\bminster\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "tower",
        "label": "tower",
        "positive_patterns": [r"\btower\b", r"\bspire\b", r"\bbelfry\b", r"\bclock tower\b", r"\bbell tower\b", r"\bminaret\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "bell tower",
        "label": "bell_tower",
        "positive_patterns": [r"\bbell tower\b", r"\bbelfry\b", r"\bcampanile\b", r"\bclock tower\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "dome",
        "label": "dome",
        "positive_patterns": [r"\bdome\b", r"\bdomed\b", r"\bcupola\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "castle",
        "label": "castle",
        "positive_patterns": [r"\bcastle\b", r"\bfortress\b", r"\bcitadel\b", r"\bschloss\b", r"\bchateau\b", r"\bchâteau\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "palace",
        "label": "palace",
        "positive_patterns": [r"\bpalace\b", r"\bpalazzo\b", r"\broyal residence\b", r"\bschloss\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "monument",
        "label": "monument",
        "positive_patterns": [r"\bmonument\b", r"\bmemorial\b", r"\bobelisk\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "statue",
        "label": "statue",
        "positive_patterns": [r"\bstatue\b", r"\bsculpture\b", r"\bequestrian statue\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "fountain",
        "label": "fountain",
        "positive_patterns": [r"\bfountain\b", r"\bfontana\b", r"\bbrunnen\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "arch",
        "label": "arch",
        "positive_patterns": [r"\barch\b", r"\barched\b", r"\btriumphal arch\b", r"\barchway\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "facade",
        "label": "facade",
        "positive_patterns": [r"\bfacade\b", r"\bfaçade\b", r"\bfrontage\b", r"\bfront facade\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "gate",
        "label": "gate",
        "positive_patterns": [r"\bgate\b", r"\bgateway\b", r"\bcity gate\b", r"\bentrance gate\b", r"\bporta\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "train",
        "label": "train",
        "positive_patterns": [r"\btrain\b", r"\blocomotive\b", r"\brailcar\b", r"\brailway\b", r"\brailroad\b", r"\btrain station\b", r"\brailway station\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "railway",
        "label": "railway",
        "positive_patterns": [r"\brailway\b", r"\brailroad\b", r"\brail line\b", r"\btracks\b", r"\btrain tracks\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "tram",
        "label": "tram",
        "positive_patterns": [r"\btram\b", r"\bstreetcar\b", r"\btrolley\b", r"\btramway\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "boat",
        "label": "boat",
        "positive_patterns": [r"\bboat\b", r"\bboats\b", r"\bferry\b", r"\bgondola\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "ship",
        "label": "ship",
        "positive_patterns": [r"\bship\b", r"\bsteamship\b", r"\bvessel\b", r"\bships\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "port",
        "label": "port",
        "positive_patterns": [r"\bport\b", r"\bharbor\b", r"\bharbour\b", r"\bdock\b", r"\bquay\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "river",
        "label": "river",
        "positive_patterns": [r"\briver\b", r"\briverside\b", r"\briverbank\b", r"\bcanal\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "shore",
        "label": "shore",
        "positive_patterns": [r"\bshore\b", r"\bshoreline\b", r"\bcoast\b", r"\bcoastline\b", r"\bseashore\b", r"\bbeach\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "mountain",
        "label": "mountain",
        "positive_patterns": [r"\bmountain\b", r"\bmountains\b", r"\balps\b", r"\bpeak\b", r"\bsummit\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "hill",
        "label": "hill",
        "positive_patterns": [r"\bhill\b", r"\bhills\b", r"\bhillside\b", r"\bhilly\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "garden",
        "label": "garden",
        "positive_patterns": [r"\bgarden\b", r"\bgardens\b", r"\bpublic garden\b", r"\bbotanical garden\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "park",
        "label": "park",
        "positive_patterns": [r"\bpark\b", r"\bparks\b", r"\bgarden\b", r"\bpublic garden\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "people",
        "label": "people",
        "positive_patterns": [r"\bpeople\b", r"\bpersons\b", r"\bmen\b", r"\bwomen\b", r"\bpedestrians\b", r"\bfigures\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "crowd",
        "label": "crowd",
        "positive_patterns": [r"\bcrowd\b", r"\bcrowded\b", r"\bgathering\b", r"\bgroup of people\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "parade",
        "label": "parade",
        "positive_patterns": [r"\bparade\b", r"\bprocession\b", r"\bmarch\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "old postcard",
        "label": "old_postcard",
        "positive_patterns": [r"\bpostcard\b", r"\bold postcard\b", r"\bvintage postcard\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "aerial view",
        "label": "aerial_view",
        "positive_patterns": [r"\baerial view\b", r"\bbird'?s[- ]eye view\b", r"\bview from above\b"],
    },
]


EUROPEANA_QUERY_GROUPS = [
    {
        "benchmark_type": "public_visual",
        "query": "castle",
        "label": "castle",
        "positive_patterns": [r"\bcastle\b", r"\bfortress\b", r"\bcitadel\b", r"\bschloss\b", r"\bchateau\b", r"\bchateau\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "church",
        "label": "church",
        "positive_patterns": [r"\bchurch\b", r"\bcathedral\b", r"\bchapel\b", r"\bbasilica\b", r"\babbey\b", r"\bkirche\b", r"\bduomo\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "bridge",
        "label": "bridge",
        "positive_patterns": [r"\bbridge\b", r"\bdrawbridge\b", r"\bviaduct\b", r"\bfootbridge\b", r"\bponte\b", r"\bpont\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "river",
        "label": "river",
        "positive_patterns": [r"\briver\b", r"\briverside\b", r"\briverbank\b", r"\bcanal\b", r"\bwaterway\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "boat",
        "label": "boat",
        "positive_patterns": [r"\bboat\b", r"\bboats\b", r"\bferry\b", r"\bgondola\b", r"\bship\b", r"\bvessel\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "fountain",
        "label": "fountain",
        "positive_patterns": [r"\bfountain\b", r"\bfontana\b", r"\bbrunnen\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "street",
        "label": "street",
        "positive_patterns": [r"\bstreet\b", r"\bstreetscape\b", r"\bavenue\b", r"\broad\b", r"\bboulevard\b", r"\blane\b", r"\bstrasse\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "tower",
        "label": "tower",
        "positive_patterns": [r"\btower\b", r"\bspire\b", r"\bbelfry\b", r"\bclock tower\b", r"\bbell tower\b"],
    },
    {
        "benchmark_type": "public_scene",
        "query": "old town street",
        "label": "old_town_street",
        "required_patterns": [
            [r"\bold town\b", r"\btown\b", r"\bcity\b", r"\bplace\b"],
            [r"\bstreet\b", r"\bstreetscape\b", r"\bavenue\b", r"\broad\b"],
        ],
    },
    {
        "benchmark_type": "public_scene",
        "query": "boats on water",
        "label": "boats_on_water",
        "required_patterns": [
            [r"\bboat\b", r"\bboats\b", r"\bship\b", r"\bferry\b", r"\bgondola\b"],
            [r"\bwater\b", r"\briver\b", r"\bcanal\b", r"\blake\b", r"\bsea\b", r"\bharbor\b", r"\bharbour\b"],
        ],
    },
    {
        "benchmark_type": "public_scene",
        "query": "castle on a hill",
        "label": "castle_on_hill",
        "required_patterns": [
            [r"\bcastle\b", r"\bfortress\b", r"\bschloss\b", r"\bchateau\b"],
            [r"\bhill\b", r"\bhillside\b", r"\bmountain\b", r"\bpeak\b"],
        ],
    },
    {
        "benchmark_type": "public_scene",
        "query": "fountain in a square",
        "label": "fountain_in_square",
        "required_patterns": [
            [r"\bfountain\b", r"\bfontana\b", r"\bbrunnen\b"],
            [r"\bsquare\b", r"\bplaza\b", r"\bpiazza\b", r"\bplatz\b", r"\bmarket square\b"],
        ],
    },
    {
        "benchmark_type": "public_scene",
        "query": "bridge over river",
        "label": "bridge_over_river",
        "required_patterns": [
            [r"\bbridge\b", r"\bdrawbridge\b", r"\bviaduct\b", r"\bponte\b", r"\bpont\b"],
            [r"\briver\b", r"\briverside\b", r"\bcanal\b", r"\bwaterway\b"],
        ],
    },
    {
        "benchmark_type": "public_scene",
        "query": "church in a square",
        "label": "church_in_square",
        "required_patterns": [
            [r"\bchurch\b", r"\bcathedral\b", r"\bchapel\b", r"\bduomo\b"],
            [r"\bsquare\b", r"\bplaza\b", r"\bpiazza\b", r"\bplatz\b"],
        ],
    },
    {
        "benchmark_type": "research_architecture",
        "query": "church tower",
        "label": "church_tower",
        "required_patterns": [
            [r"\bchurch\b", r"\bcathedral\b", r"\bchapel\b", r"\bduomo\b"],
            [r"\btower\b", r"\bspire\b", r"\bbell tower\b", r"\bbelfry\b"],
        ],
    },
    {
        "benchmark_type": "research_architecture",
        "query": "dome",
        "label": "dome",
        "positive_patterns": [r"\bdome\b", r"\bdomed\b", r"\bcupola\b"],
    },
    {
        "benchmark_type": "research_architecture",
        "query": "arched window",
        "label": "arched_window",
        "required_patterns": [
            [r"\barch\b", r"\barched\b", r"\barchway\b"],
            [r"\bwindow\b", r"\bwindows\b"],
        ],
    },
    {
        "benchmark_type": "research_architecture",
        "query": "building facade",
        "label": "building_facade",
        "positive_patterns": [r"\bfacade\b", r"\bfrontage\b", r"\bfront facade\b", r"\bbuilding facade\b"],
    },
    {
        "benchmark_type": "research_architecture",
        "query": "statue on pedestal",
        "label": "statue_on_pedestal",
        "required_patterns": [
            [r"\bstatue\b", r"\bsculpture\b", r"\bmonument\b"],
            [r"\bpedestal\b", r"\bbase\b", r"\bplinth\b"],
        ],
    },
    {
        "benchmark_type": "research_architecture",
        "query": "bell tower",
        "label": "bell_tower",
        "positive_patterns": [r"\bbell tower\b", r"\bbelfry\b", r"\bcampanile\b", r"\bclock tower\b"],
    },
    {
        "benchmark_type": "research_spatial",
        "query": "market square",
        "label": "market_square",
        "required_patterns": [
            [r"\bmarket\b", r"\bmarketplace\b", r"\bfair\b"],
            [r"\bsquare\b", r"\bplaza\b", r"\bpiazza\b", r"\bplatz\b"],
        ],
    },
    {
        "benchmark_type": "research_spatial",
        "query": "waterfront promenade",
        "label": "waterfront_promenade",
        "positive_patterns": [r"\bwaterfront\b", r"\briverside\b", r"\bseafront\b", r"\bpromenade\b", r"\bquay\b", r"\bembankment\b"],
    },
    {
        "benchmark_type": "research_spatial",
        "query": "harbor with boats",
        "label": "harbor_with_boats",
        "required_patterns": [
            [r"\bharbor\b", r"\bharbour\b", r"\bport\b", r"\bdock\b", r"\bquay\b"],
            [r"\bboat\b", r"\bboats\b", r"\bship\b", r"\bvessel\b"],
        ],
    },
    {
        "benchmark_type": "research_spatial",
        "query": "riverfront with buildings",
        "label": "riverfront_with_buildings",
        "required_patterns": [
            [r"\briverfront\b", r"\briverside\b", r"\briver\b", r"\bcanal\b"],
            [r"\bbuilding\b", r"\bbuildings\b", r"\bfacade\b", r"\bhouse\b"],
        ],
    },
    {
        "benchmark_type": "research_spatial",
        "query": "arcaded street",
        "label": "arcaded_street",
        "required_patterns": [
            [r"\barcade\b", r"\barcaded\b", r"\barches\b", r"\barchway\b"],
            [r"\bstreet\b", r"\bavenue\b", r"\broad\b"],
        ],
    },
    {
        "benchmark_type": "research_spatial",
        "query": "railway station",
        "label": "railway_station",
        "positive_patterns": [r"\brailway station\b", r"\btrain station\b", r"\bbahnhof\b", r"\bstation\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "square",
        "label": "square",
        "positive_patterns": [r"\bsquare\b", r"\bplaza\b", r"\bpiazza\b", r"\bplatz\b", r"\bmarket square\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "market",
        "label": "market",
        "positive_patterns": [r"\bmarket\b", r"\bmarketplace\b", r"\bfair\b", r"\bbazaar\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "harbor",
        "label": "harbor",
        "positive_patterns": [r"\bharbor\b", r"\bharbour\b", r"\bport\b", r"\bdock\b", r"\bquay\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "canal",
        "label": "canal",
        "positive_patterns": [r"\bcanal\b", r"\bwaterway\b", r"\bchannel\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "train",
        "label": "train",
        "positive_patterns": [r"\btrain\b", r"\blocomotive\b", r"\brailway\b", r"\brailroad\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "tram",
        "label": "tram",
        "positive_patterns": [r"\btram\b", r"\bstreetcar\b", r"\btrolley\b", r"\btramway\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "monument",
        "label": "monument",
        "positive_patterns": [r"\bmonument\b", r"\bmemorial\b", r"\bobelisk\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "statue",
        "label": "statue",
        "positive_patterns": [r"\bstatue\b", r"\bsculpture\b", r"\bequestrian statue\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "garden",
        "label": "garden",
        "positive_patterns": [r"\bgarden\b", r"\bgardens\b", r"\bpublic garden\b", r"\bbotanical garden\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "park",
        "label": "park",
        "positive_patterns": [r"\bpark\b", r"\bparks\b", r"\bgarden\b", r"\bpublic garden\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "mountain",
        "label": "mountain",
        "positive_patterns": [r"\bmountain\b", r"\bmountains\b", r"\balps\b", r"\bpeak\b", r"\bsummit\b"],
    },
    {
        "benchmark_type": "public_visual",
        "query": "people",
        "label": "people",
        "positive_patterns": [r"\bpeople\b", r"\bpersons\b", r"\bmen\b", r"\bwomen\b", r"\bpedestrians\b", r"\bfigures\b"],
    },
    {
        "benchmark_type": "public_scene",
        "query": "street with tram",
        "label": "street_with_tram",
        "required_patterns": [
            [r"\bstreet\b", r"\bavenue\b", r"\broad\b"],
            [r"\btram\b", r"\bstreetcar\b", r"\btrolley\b"],
        ],
    },
    {
        "benchmark_type": "public_scene",
        "query": "people in street",
        "label": "people_in_street",
        "required_patterns": [
            [r"\bpeople\b", r"\bpersons\b", r"\bpedestrians\b", r"\bmen\b", r"\bwomen\b"],
            [r"\bstreet\b", r"\bavenue\b", r"\broad\b"],
        ],
    },
    {
        "benchmark_type": "public_scene",
        "query": "monument in square",
        "label": "monument_in_square",
        "required_patterns": [
            [r"\bmonument\b", r"\bmemorial\b", r"\bstatue\b"],
            [r"\bsquare\b", r"\bplaza\b", r"\bpiazza\b", r"\bplatz\b"],
        ],
    },
    {
        "benchmark_type": "public_scene",
        "query": "train station",
        "label": "train_station",
        "required_patterns": [
            [r"\btrain\b", r"\brailway\b", r"\brailroad\b"],
            [r"\bstation\b", r"\bbahnhof\b"],
        ],
    },
    {
        "benchmark_type": "public_scene",
        "query": "ship in harbor",
        "label": "ship_in_harbor",
        "required_patterns": [
            [r"\bship\b", r"\bvessel\b", r"\bboat\b"],
            [r"\bharbor\b", r"\bharbour\b", r"\bport\b", r"\bdock\b"],
        ],
    },
    {
        "benchmark_type": "public_scene",
        "query": "mountain landscape",
        "label": "mountain_landscape",
        "required_patterns": [
            [r"\bmountain\b", r"\bmountains\b", r"\balps\b"],
            [r"\blandscape\b", r"\bview\b", r"\bpanorama\b"],
        ],
    },
    {
        "benchmark_type": "research_architecture",
        "query": "city gate",
        "label": "city_gate",
        "positive_patterns": [r"\bgate\b", r"\bgateway\b", r"\bcity gate\b", r"\bporta\b"],
    },
    {
        "benchmark_type": "research_architecture",
        "query": "arched doorway",
        "label": "arched_doorway",
        "required_patterns": [
            [r"\barch\b", r"\barched\b", r"\barchway\b"],
            [r"\bdoor\b", r"\bdoorway\b", r"\bentrance\b"],
        ],
    },
    {
        "benchmark_type": "research_architecture",
        "query": "clock tower",
        "label": "clock_tower",
        "positive_patterns": [r"\bclock tower\b", r"\btower clock\b"],
    },
    {
        "benchmark_type": "research_architecture",
        "query": "castle walls",
        "label": "castle_walls",
        "required_patterns": [
            [r"\bcastle\b", r"\bfortress\b", r"\bschloss\b"],
            [r"\bwall\b", r"\bwalls\b", r"\bfortification\b"],
        ],
    },
    {
        "benchmark_type": "research_architecture",
        "query": "palace facade",
        "label": "palace_facade",
        "required_patterns": [
            [r"\bpalace\b", r"\bpalazzo\b", r"\bschloss\b"],
            [r"\bfacade\b", r"\bfrontage\b", r"\bfront\b"],
        ],
    },
    {
        "benchmark_type": "research_spatial",
        "query": "city square",
        "label": "city_square",
        "required_patterns": [
            [r"\bcity\b", r"\btown\b", r"\bplace\b"],
            [r"\bsquare\b", r"\bplaza\b", r"\bpiazza\b", r"\bplatz\b"],
        ],
    },
    {
        "benchmark_type": "research_spatial",
        "query": "canal with buildings",
        "label": "canal_with_buildings",
        "required_patterns": [
            [r"\bcanal\b", r"\bwaterway\b"],
            [r"\bbuilding\b", r"\bbuildings\b", r"\bhouse\b"],
        ],
    },
    {
        "benchmark_type": "research_spatial",
        "query": "street market",
        "label": "street_market",
        "required_patterns": [
            [r"\bstreet\b", r"\bavenue\b", r"\broad\b"],
            [r"\bmarket\b", r"\bmarketplace\b", r"\bfair\b"],
        ],
    },
    {
        "benchmark_type": "research_spatial",
        "query": "park promenade",
        "label": "park_promenade",
        "required_patterns": [
            [r"\bpark\b", r"\bgarden\b"],
            [r"\bpromenade\b", r"\bwalk\b", r"\bavenue\b"],
        ],
    },
    {
        "benchmark_type": "research_spatial",
        "query": "railway bridge",
        "label": "railway_bridge",
        "required_patterns": [
            [r"\brailway\b", r"\brailroad\b", r"\btrain\b"],
            [r"\bbridge\b", r"\bviaduct\b"],
        ],
    },
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "in",
    "of",
    "on",
    "over",
    "the",
    "to",
    "with",
}


def fix_text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        repaired = text.encode("latin1").decode("utf-8")
        bad_original = text.count("Ã") + text.count("Â")
        bad_repaired = repaired.count("Ã") + repaired.count("Â")
        if bad_repaired < bad_original:
            return repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text


def normalize_negative_prompts(negative_text):
    prompts = []
    if negative_text:
        prompts = [part.strip() for part in negative_text.split(",") if part.strip()]
    return prompts or ["background"]


def precision_at_k(items, relevant_set, k):
    top_items = items[:k]
    if not top_items:
        return 0.0
    hits = sum(1 for item in top_items if item in relevant_set)
    return hits / float(k)


def recall_at_k(items, relevant_set, k):
    if not relevant_set:
        return 0.0
    hits = sum(1 for item in items[:k] if item in relevant_set)
    return hits / float(len(relevant_set))


def reciprocal_rank(items, relevant_set):
    for idx, item in enumerate(items, start=1):
        if item in relevant_set:
            return 1.0 / float(idx)
    return 0.0


def average_precision(items, relevant_set, k=None):
    if not relevant_set:
        return 0.0
    ranked = items if k is None else items[:k]
    hit_count = 0
    precisions = []
    for idx, item in enumerate(ranked, start=1):
        if item in relevant_set:
            hit_count += 1
            precisions.append(hit_count / float(idx))
    if not precisions:
        return 0.0
    return sum(precisions) / float(len(relevant_set))


def ndcg_at_k(items, relevant_set, k):
    if not relevant_set:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(items[:k], start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_set), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def slugify_for_filename(text):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "benchmark"


def query_family(benchmark_type):
    if benchmark_type.startswith("public_"):
        return "public"
    if benchmark_type.startswith("research_"):
        return "research"
    if benchmark_type.startswith("named_") or benchmark_type in {"city", "place", "landmark"}:
        return "named"
    return "general" if benchmark_type == "concept" else "specific"


def query_group_label(benchmark_type):
    labels = {
        "public_visual": "Public visual concepts",
        "public_scene": "Public scene queries",
        "research_architecture": "Research architectural queries",
        "research_spatial": "Research spatial queries",
        "named_place": "Named place queries",
        "named_landmark": "Named landmark queries",
        "city": "City queries",
        "place": "Place queries",
        "landmark": "Landmark queries",
        "concept": "Legacy concept queries",
    }
    return labels.get(benchmark_type, benchmark_type)


def build_name_lexicon(df):
    names = set()
    for column in ("final_city_clean", "final_place_clean"):
        if column not in df.columns:
            continue
        for value in df[column].dropna().astype(str):
            value = value.strip().lower()
            if len(value) >= 3:
                names.add(value)
    for names_list in df.get("landmark_names", []):
        for value in names_list:
            value = str(value).strip().lower()
            if len(value) >= 3:
                names.add(value)
    return names


def query_specificity_score(query):
    tokens = [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", str(query).lower())
        if token not in STOPWORDS
    ]
    if not tokens:
        return 0.0
    return min(1.0, max(0.0, (len(tokens) - 1) / 4.0))


def query_name_score(query, name_lexicon):
    query_text = str(query).strip().lower()
    if not query_text or not name_lexicon:
        return 0.0
    if query_text in name_lexicon:
        return 1.0
    compact_query = re.sub(r"\s+", " ", query_text)
    for name in name_lexicon:
        if len(name) < 4:
            continue
        if name in compact_query or compact_query in name:
            return 1.0
    return 0.0


def dynamic_metadata_weight(query, benchmark_type, name_lexicon, base_weight):
    if benchmark_type.startswith("named_") or benchmark_type in {"city", "place", "landmark"}:
        name_score = 1.0
    else:
        name_score = query_name_score(query, name_lexicon)
    spec_score = query_specificity_score(query)
    return float(min(1.0, max(0.0, 0.15 + 0.55 * name_score + 0.15 * spec_score)))


def compute_query_weights(query_spec, args, name_lexicon):
    metadata_weight = float(args.metadata_weight)
    if args.dynamic_metadata_weight:
        metadata_weight = dynamic_metadata_weight(
            query_spec["query"],
            query_spec["benchmark_type"],
            name_lexicon,
            base_weight=args.metadata_weight,
        )
    return {
        "cluster_weight": float(args.cluster_weight),
        "cls_weight": float(args.cls_weight),
        "metadata_weight": metadata_weight,
    }


def escape_html(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return html.escape(str(value))


def extract_landmark_names(value):
    text = fix_text(value)
    if not text:
        return []
    matches = re.findall(r"\s*([^,(][^()]*)\s*\(https?://[^)]+\)", text)
    if matches:
        return [match.strip(" ,") for match in matches if match.strip(" ,")]
    return []


def build_combined_text(row):
    parts = [
        row.get("description_clean", ""),
        row.get("transcription_clean", ""),
        row.get("landmarks_clean", ""),
        row.get("final_place_clean", ""),
        row.get("final_city_clean", ""),
        row.get("final_country_clean", ""),
    ]
    return " ".join(part for part in parts if part).lower()


class SearchScorer:
    def __init__(
        self,
        backend_name,
        es_host,
        es_index,
        device,
        es_timeout=60.0,
        model_id=None,
        model_version="c-radio_v4-h",
        lang_model="siglip2-g",
        vector_field="vector",
        image_id_field="image_id",
        cluster_id_field="cluster_id",
        candidate_k=120,
        negative_text="background, sky, clouds, text, border, trees, road, people",
        temperature=10.0,
        cluster_weight=1.0,
        cls_weight=0.2,
        metadata_weight=0.3,
        image_aggregation="max",
        image_agg_top_k=3,
        image_agg_alpha=0.7,
    ):
        self.es = Elasticsearch(es_host, request_timeout=es_timeout)
        self.es_index = es_index
        self.vector_field = vector_field
        self.image_id_field = image_id_field
        self.cluster_id_field = cluster_id_field
        self.embedding_type_field = "embedding_type"
        self.candidate_k = candidate_k
        self.temperature = float(temperature)
        self.negative_prompts = normalize_negative_prompts(negative_text)
        self.cluster_weight = float(cluster_weight)
        self.cls_weight = float(cls_weight)
        self.metadata_weight = float(metadata_weight)
        self.image_aggregation = image_aggregation
        self.image_agg_top_k = int(image_agg_top_k)
        self.image_agg_alpha = float(image_agg_alpha)

        self.backend = create_backend(
            backend_name=backend_name,
            device=device,
            model_id=model_id,
            model_version=model_version,
            lang_model=lang_model,
        )

    def aggregate_image_score(self, cluster_hits):
        if not cluster_hits:
            return 0.0
        scores = sorted((float(hit["score"]) for hit in cluster_hits), reverse=True)
        max_score = scores[0]
        if self.image_aggregation == "max":
            return max_score
        if self.image_aggregation == "topk_mean":
            k = max(1, min(self.image_agg_top_k, len(scores)))
            return float(np.mean(scores[:k]))
        if self.image_aggregation == "max_topk_mean":
            k = max(1, min(self.image_agg_top_k, len(scores)))
            topk_mean = float(np.mean(scores[:k]))
            alpha = float(np.clip(self.image_agg_alpha, 0.0, 1.0))
            return alpha * max_score + (1.0 - alpha) * topk_mean
        raise ValueError(f"Unsupported image aggregation: {self.image_aggregation}")

    @torch.no_grad()
    def encode_prompts(self, prompts):
        embeddings = self.backend.encode_text(prompts)
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        return F.normalize(embeddings, dim=-1)

    @staticmethod
    def empty_timing():
        return {
            "query_encoding_s": 0.0,
            "es_knn_search_s": 0.0,
            "negative_rerank_s": 0.0,
            "cls_metadata_scoring_s": 0.0,
            "image_aggregation_s": 0.0,
            "total_latency_s": 0.0,
        }

    def knn_candidates(self, query_vector):
        response = self.es.search(
            index=self.es_index,
            knn={
                "field": self.vector_field,
                "query_vector": query_vector,
                "k": self.candidate_k,
                "num_candidates": max(self.candidate_k * 4, 100),
                "filter": self.cluster_doc_query(),
            },
            _source=[self.image_id_field, self.cluster_id_field],
            size=self.candidate_k,
        )
        return response["hits"]["hits"]

    def cluster_doc_query(self):
        return {
            "bool": {
                "should": [
                    {"term": {self.embedding_type_field: "cluster"}},
                    {"bool": {"must_not": {"exists": {"field": self.embedding_type_field}}}},
                ],
                "minimum_should_match": 1,
            }
        }

    def combine_with_cluster_filter(self, query):
        return {"bool": {"filter": [query, self.cluster_doc_query()]}}

    def score_query(self, positive_vector, negative_vectors, candidate_query, size):
        script_source = f"""
double pos = cosineSimilarity(params.positive_vector, '{self.vector_field}');
double numer = Math.exp(pos * params.temperature);
double denom = numer;
double bestNeg = -1000.0;
for (neg in params.negative_vectors) {{
  double negScore = cosineSimilarity(neg, '{self.vector_field}');
  if (negScore > bestNeg) bestNeg = negScore;
  denom += Math.exp(negScore * params.temperature);
}}
double score = numer / denom;
return score;
"""

        response = self.es.search(
            index=self.es_index,
            query={
                "script_score": {
                    "query": candidate_query,
                    "script": {
                        "source": script_source,
                        "params": {
                            "positive_vector": positive_vector,
                            "negative_vectors": negative_vectors,
                            "temperature": self.temperature,
                        },
                    },
                }
            },
            _source=[self.image_id_field, self.cluster_id_field, self.embedding_type_field],
            size=size,
        )
        return response["hits"]["hits"]

    def image_auxiliary_scores(self, image_ids, positive_vector, negative_vectors):
        if not image_ids or (self.cls_weight == 0.0 and self.metadata_weight == 0.0):
            return {}
        unique_image_ids = sorted(set(image_ids))
        hits = self.score_query(
            positive_vector=positive_vector,
            negative_vectors=negative_vectors,
            candidate_query={
                "bool": {
                    "filter": [
                        {"terms": {self.image_id_field: unique_image_ids}},
                        {"terms": {self.embedding_type_field: ["cls", "metadata"]}},
                    ]
                }
            },
            size=max(len(unique_image_ids) * 2, 1),
        )
        scores = {}
        for hit in hits:
            source = hit.get("_source", {})
            image_id = source.get(self.image_id_field)
            embedding_type = source.get(self.embedding_type_field)
            if not image_id or embedding_type not in {"cls", "metadata"}:
                continue
            payload = scores.setdefault(image_id, {"cls_score": 0.0, "metadata_score": 0.0})
            key = "cls_score" if embedding_type == "cls" else "metadata_score"
            payload[key] = max(payload[key], float(hit.get("_score", 0.0)))
        return scores

    def apply_weighted_cluster_scores(self, scored_hits, auxiliary_scores, weights=None):
        weights = weights or {}
        cluster_weight = float(weights.get("cluster_weight", self.cluster_weight))
        cls_weight = float(weights.get("cls_weight", self.cls_weight))
        metadata_weight = float(weights.get("metadata_weight", self.metadata_weight))
        for item in scored_hits:
            aux = auxiliary_scores.get(item["image_id"], {})
            cluster_score = float(item["cluster_score"])
            cls_score = float(aux.get("cls_score", 0.0))
            metadata_score = float(aux.get("metadata_score", 0.0))
            item["cls_score"] = cls_score
            item["metadata_score"] = metadata_score
            item["score"] = (
                cluster_weight * cluster_score
                + cls_weight * cls_score
                + metadata_weight * metadata_score
            )
            item["cluster_weight"] = cluster_weight
            item["cls_weight"] = cls_weight
            item["metadata_weight"] = metadata_weight
        return scored_hits

    def search(self, query_text, result_mode="image", top_k=50, weights=None, return_timing=False):
        timing = self.empty_timing()
        total_start = time.perf_counter()

        prompts = [query_text] + self.negative_prompts
        stage_start = time.perf_counter()
        text_vectors = self.encode_prompts(prompts)
        timing["query_encoding_s"] = time.perf_counter() - stage_start
        positive_vector = text_vectors[0].detach().cpu().numpy().tolist()
        negative_vectors = [vec.detach().cpu().numpy().tolist() for vec in text_vectors[1:]]

        stage_start = time.perf_counter()
        preselected_hits = self.knn_candidates(positive_vector)
        timing["es_knn_search_s"] = time.perf_counter() - stage_start
        candidate_ids = [hit["_id"] for hit in preselected_hits]
        if not candidate_ids:
            timing["total_latency_s"] = time.perf_counter() - total_start
            return ([], timing) if return_timing else []

        stage_start = time.perf_counter()
        hits = self.score_query(
            positive_vector=positive_vector,
            negative_vectors=negative_vectors,
            candidate_query=self.combine_with_cluster_filter({"ids": {"values": candidate_ids}}),
            size=self.candidate_k,
        )
        timing["negative_rerank_s"] = time.perf_counter() - stage_start

        scored_hits = []
        for hit in hits:
            source = hit.get("_source", {})
            scored_hits.append(
                {
                    "image_id": source[self.image_id_field],
                    "cluster_id": int(source.get(self.cluster_id_field, 0)),
                    "score": float(hit.get("_score", 0.0)),
                    "cluster_score": float(hit.get("_score", 0.0)),
                    "cls_score": 0.0,
                    "metadata_score": 0.0,
                }
            )

        stage_start = time.perf_counter()
        auxiliary_scores = self.image_auxiliary_scores(
            image_ids=[item["image_id"] for item in scored_hits],
            positive_vector=positive_vector,
            negative_vectors=negative_vectors,
        )
        timing["cls_metadata_scoring_s"] = time.perf_counter() - stage_start
        scored_hits = self.apply_weighted_cluster_scores(scored_hits, auxiliary_scores, weights=weights)

        if result_mode == "cluster":
            stage_start = time.perf_counter()
            results = sorted(scored_hits, key=lambda item: item["score"], reverse=True)[:top_k]
            timing["image_aggregation_s"] = time.perf_counter() - stage_start
            timing["total_latency_s"] = time.perf_counter() - total_start
            return (results, timing) if return_timing else results

        stage_start = time.perf_counter()
        best_per_image = {}
        for item in scored_hits:
            image_id = item["image_id"]
            if image_id not in best_per_image:
                best_per_image[image_id] = {
                    "image_id": image_id,
                    "cluster_id": int(item["cluster_id"]),
                    "score": float(item["score"]),
                    "cluster_score": float(item.get("cluster_score", 0.0)),
                    "cls_score": float(item.get("cls_score", 0.0)),
                    "metadata_score": float(item.get("metadata_score", 0.0)),
                    "cluster_weight": float(item.get("cluster_weight", self.cluster_weight)),
                    "cls_weight": float(item.get("cls_weight", self.cls_weight)),
                    "metadata_weight": float(item.get("metadata_weight", self.metadata_weight)),
                    "cluster_hits": [item],
                }
            else:
                best_per_image[image_id]["cluster_hits"].append(item)

        for payload in best_per_image.values():
            dedup = {}
            for hit in payload["cluster_hits"]:
                cluster_id = int(hit["cluster_id"])
                if cluster_id not in dedup or hit["score"] > dedup[cluster_id]["score"]:
                    dedup[cluster_id] = hit
            payload["cluster_hits"] = sorted(
                dedup.values(),
                key=lambda hit: hit["score"],
                reverse=True,
            )
            top_hit = payload["cluster_hits"][0]
            payload["score"] = self.aggregate_image_score(payload["cluster_hits"])
            payload["cluster_id"] = int(top_hit["cluster_id"])
            payload["cluster_score"] = float(top_hit.get("cluster_score", 0.0))
            payload["cls_score"] = float(top_hit.get("cls_score", 0.0))
            payload["metadata_score"] = float(top_hit.get("metadata_score", 0.0))
            payload["cluster_weight"] = float(top_hit.get("cluster_weight", self.cluster_weight))
            payload["cls_weight"] = float(top_hit.get("cls_weight", self.cls_weight))
            payload["metadata_weight"] = float(top_hit.get("metadata_weight", self.metadata_weight))
            payload["image_aggregation"] = self.image_aggregation
            payload["image_agg_top_k"] = self.image_agg_top_k
            payload["image_agg_alpha"] = self.image_agg_alpha

        results = sorted(best_per_image.values(), key=lambda item: item["score"], reverse=True)[:top_k]
        timing["image_aggregation_s"] = time.perf_counter() - stage_start
        timing["total_latency_s"] = time.perf_counter() - total_start
        return (results, timing) if return_timing else results


def get_indexed_image_ids(es, index_name):
    image_ids = set()
    for hit in helpers.scan(
        client=es,
        index=index_name,
        query={"query": {"match_all": {}}},
        _source=["image_id"],
        size=1000,
    ):
        image_ids.add(hit["_source"]["image_id"])
    return image_ids


def prepare_metadata(metadata_csv, indexed_image_ids):
    df = pd.read_csv(metadata_csv, encoding="latin1")
    df["image_filename"] = df["image_filename"].map(fix_text)
    df["final_country_clean"] = df["final_country"].map(fix_text)
    df["final_city_clean"] = df["final_city"].map(fix_text)
    df["final_place_clean"] = df["final_place"].map(fix_text)
    df["description_clean"] = df["description"].map(fix_text)
    df["transcription_clean"] = df["transcription"].map(fix_text)
    df["landmarks_clean"] = df["landmarks_identified"].map(fix_text)
    df["landmark_names"] = df["landmarks_identified"].map(extract_landmark_names)
    df["combined_text"] = df.apply(build_combined_text, axis=1)
    df = df[df["image_filename"].isin(indexed_image_ids)].copy()
    return df


def build_structured_queries(
    df,
    min_city_count,
    min_place_count,
    min_landmark_count,
    top_n_city,
    top_n_place,
    top_n_landmark,
):
    query_specs = []

    city_counts = df["final_city_clean"].value_counts()
    city_counts = city_counts[city_counts.index.str.strip() != ""]
    for value, count in city_counts[city_counts >= min_city_count].head(top_n_city).items():
        relevant = set(df.loc[df["final_city_clean"] == value, "image_filename"])
        query_specs.append(
            {
                "benchmark_type": "city",
                "query": value,
                "label": value,
                "relevant_images": relevant,
                "support": len(relevant),
            }
        )

    place_counts = df["final_place_clean"].value_counts()
    place_counts = place_counts[place_counts.index.str.strip() != ""]
    for value, count in place_counts[place_counts >= min_place_count].head(top_n_place).items():
        relevant = set(df.loc[df["final_place_clean"] == value, "image_filename"])
        query_specs.append(
            {
                "benchmark_type": "place",
                "query": value,
                "label": value,
                "relevant_images": relevant,
                "support": len(relevant),
            }
        )

    landmark_counter = Counter()
    landmark_to_images = defaultdict(set)
    for _, row in df.iterrows():
        image_id = row["image_filename"]
        for landmark_name in row["landmark_names"]:
            if not landmark_name:
                continue
            landmark_counter[landmark_name] += 1
            landmark_to_images[landmark_name].add(image_id)

    if top_n_landmark <= 0:
        return query_specs

    landmark_count = 0
    for landmark_name, count in landmark_counter.most_common():
        if count < min_landmark_count:
            break
        if landmark_count >= top_n_landmark:
            break
        query_specs.append(
            {
                "benchmark_type": "landmark",
                "query": landmark_name,
                "label": landmark_name,
                "relevant_images": landmark_to_images[landmark_name],
                "support": len(landmark_to_images[landmark_name]),
            }
        )
        landmark_count += 1

    return query_specs


def query_mask_from_spec(text_series, spec):
    if "required_patterns" in spec:
        mask = pd.Series(True, index=text_series.index)
        for pattern_group in spec["required_patterns"]:
            group_pattern = "|".join(pattern_group)
            mask = mask & text_series.str.contains(group_pattern, regex=True)
        return mask
    pattern = "|".join(spec["positive_patterns"])
    return text_series.str.contains(pattern, regex=True)


def build_named_europeana_queries(
    df,
    min_place_count,
    min_landmark_count,
    top_n_place,
    top_n_landmark,
):
    query_specs = []

    place_counts = df["final_place_clean"].value_counts()
    place_counts = place_counts[place_counts.index.str.strip() != ""]
    for value, count in place_counts[place_counts >= min_place_count].head(top_n_place).items():
        relevant = set(df.loc[df["final_place_clean"] == value, "image_filename"])
        query_specs.append(
            {
                "benchmark_type": "named_place",
                "query": value,
                "label": value,
                "relevant_images": relevant,
                "support": len(relevant),
            }
        )

    landmark_counter = Counter()
    landmark_to_images = defaultdict(set)
    for _, row in df.iterrows():
        image_id = row["image_filename"]
        for landmark_name in row["landmark_names"]:
            if not landmark_name:
                continue
            landmark_counter[landmark_name] += 1
            landmark_to_images[landmark_name].add(image_id)

    landmark_count = 0
    for landmark_name, count in landmark_counter.most_common():
        if count < min_landmark_count:
            break
        if landmark_count >= top_n_landmark:
            break
        query_specs.append(
            {
                "benchmark_type": "named_landmark",
                "query": landmark_name,
                "label": landmark_name,
                "relevant_images": landmark_to_images[landmark_name],
                "support": len(landmark_to_images[landmark_name]),
            }
        )
        landmark_count += 1

    return query_specs


def build_europeana_queries(
    df,
    min_place_count,
    min_landmark_count,
    top_n_place,
    top_n_landmark,
):
    query_specs = []
    text = df["combined_text"]
    for spec in EUROPEANA_QUERY_GROUPS:
        mask = query_mask_from_spec(text, spec)
        relevant = set(df.loc[mask, "image_filename"])
        query_specs.append(
            {
                "benchmark_type": spec["benchmark_type"],
                "query": spec["query"],
                "label": spec["label"],
                "relevant_images": relevant,
                "support": len(relevant),
            }
        )

    query_specs.extend(
        build_named_europeana_queries(
            df,
            min_place_count=min_place_count,
            min_landmark_count=min_landmark_count,
            top_n_place=top_n_place,
            top_n_landmark=top_n_landmark,
        )
    )
    return query_specs


def build_concept_queries(df):
    query_specs = []
    text = df["combined_text"]

    for concept in COMMON_CONCEPTS:
        mask = query_mask_from_spec(text, concept)

        relevant = set(df.loc[mask, "image_filename"])
        query_specs.append(
            {
                "benchmark_type": concept["benchmark_type"],
                "query": concept["query"],
                "label": concept["label"],
                "relevant_images": relevant,
                "support": len(relevant),
            }
        )

    return query_specs


def evaluate_image_level(query_spec, image_results):
    ranked_images = [item["image_id"] for item in image_results]
    relevant = query_spec["relevant_images"]
    return {
        "precision@1": precision_at_k(ranked_images, relevant, 1),
        "precision@5": precision_at_k(ranked_images, relevant, 5),
        "precision@10": precision_at_k(ranked_images, relevant, 10),
        "recall@1": recall_at_k(ranked_images, relevant, 1),
        "recall@5": recall_at_k(ranked_images, relevant, 5),
        "recall@10": recall_at_k(ranked_images, relevant, 10),
        "mrr": reciprocal_rank(ranked_images, relevant),
        "ap@10": average_precision(ranked_images, relevant, k=10),
        "ndcg@10": ndcg_at_k(ranked_images, relevant, 10),
    }


def evaluate_cluster_level(query_spec, cluster_results):
    relevant = query_spec["relevant_images"]
    ranked_cluster_images = [item["image_id"] for item in cluster_results]
    unique_ranked_images = []
    seen = set()
    for image_id in ranked_cluster_images:
        if image_id not in seen:
            seen.add(image_id)
            unique_ranked_images.append(image_id)

    top10 = cluster_results[:10]
    top20 = cluster_results[:20]
    rel_top10 = [item for item in top10 if item["image_id"] in relevant]
    rel_top20 = [item for item in top20 if item["image_id"] in relevant]
    top20_image_counts = Counter(item["image_id"] for item in rel_top20)
    repeated_relevant_images = sum(1 for _, count in top20_image_counts.items() if count >= 2)

    return {
        "cluster_precision@10": len(rel_top10) / 10.0 if top10 else 0.0,
        "cluster_precision@20": len(rel_top20) / 20.0 if top20 else 0.0,
        "image_recall_from_clusters@10": recall_at_k(unique_ranked_images, relevant, 10),
        "image_recall_from_clusters@20": recall_at_k(unique_ranked_images, relevant, 20),
        "cluster_mrr": reciprocal_rank(ranked_cluster_images, relevant),
        "relevant_unique_images@20": len(top20_image_counts),
        "repeated_relevant_images@20": repeated_relevant_images,
        "mean_relevant_cluster_score@20": float(np.mean([item["score"] for item in rel_top20])) if rel_top20 else 0.0,
        "mean_nonrelevant_cluster_score@20": float(np.mean([item["score"] for item in top20 if item["image_id"] not in relevant])) if top20 else 0.0,
    }


def export_ground_truth(report_dir, query_specs, metadata_df):
    metadata_lookup = metadata_df.set_index("image_filename")
    gt_json_path = os.path.join(report_dir, "query_ground_truth.json")
    gt_csv_path = os.path.join(report_dir, "query_ground_truth_pairs.csv")

    exported_queries = []
    gt_rows = []

    for query_spec in sorted(query_specs, key=lambda item: (item["benchmark_type"], item["query"])):
        relevant_images = sorted(query_spec["relevant_images"])
        relevant_items = []
        for image_id in relevant_images:
            if image_id in metadata_lookup.index:
                row = metadata_lookup.loc[image_id]
                final_city = row["final_city_clean"]
                final_place = row["final_place_clean"]
                landmarks_identified = row["landmarks_clean"]
                description = row["description_clean"]
            else:
                final_city = ""
                final_place = ""
                landmarks_identified = ""
                description = ""

            item = {
                "image_id": image_id,
                "final_city": final_city,
                "final_place": final_place,
                "landmarks_identified": landmarks_identified,
                "description": description,
            }
            relevant_items.append(item)

            gt_rows.append(
                {
                    "benchmark_type": query_spec["benchmark_type"],
                    "query_family": query_family(query_spec["benchmark_type"]),
                    "query": query_spec["query"],
                    "label": query_spec["label"],
                    "support": query_spec["support"],
                    "image_id": image_id,
                    "final_city": final_city,
                    "final_place": final_place,
                    "landmarks_identified": landmarks_identified,
                    "description": description,
                }
            )

        exported_queries.append(
            {
                "benchmark_type": query_spec["benchmark_type"],
                "query_family": query_family(query_spec["benchmark_type"]),
                "query": query_spec["query"],
                "label": query_spec["label"],
                "support": query_spec["support"],
                "relevant_images": relevant_images,
                "relevant_items": relevant_items,
            }
        )

    with open(gt_json_path, "w", encoding="utf-8") as handle:
        json.dump(exported_queries, handle, ensure_ascii=False, indent=2)

    pd.DataFrame(gt_rows).to_csv(gt_csv_path, index=False, encoding="utf-8")
    return gt_json_path, gt_csv_path


def write_query_browser_report(report_dir, image_df, cluster_df, gt_pairs_df, image_hits_df, cluster_hits_df, visualizations_dir, config=None):
    md_path = os.path.join(report_dir, "query_browser_report.md")
    html_path = os.path.join(report_dir, "query_browser_report.html")
    config = config or {}

    image_lookup = {
        (row["benchmark_type"], row["query"]): row
        for _, row in image_df.iterrows()
    }
    cluster_lookup = {
        (row["benchmark_type"], row["query"]): row
        for _, row in cluster_df.iterrows()
    }

    query_keys = sorted(image_lookup.keys(), key=lambda item: (query_family(item[0]), item[0], item[1].lower()))

    overview_rows = []
    for family in sorted(image_df["query_family"].unique()):
        image_subset = image_df[image_df["query_family"] == family] if "query_family" in image_df.columns else image_df[image_df["benchmark_type"].map(query_family) == family]
        cluster_subset = cluster_df[cluster_df["query_family"] == family] if "query_family" in cluster_df.columns else cluster_df[cluster_df["benchmark_type"].map(query_family) == family]
        if image_subset.empty:
            continue
        overview_rows.append(
            {
                "family": family,
                "queries": len(image_subset),
                "image_recall@5": image_subset["recall@5"].mean(),
                "image_recall@10": image_subset["recall@10"].mean(),
                "image_mrr": image_subset["mrr"].mean(),
                "image_ap@10": image_subset["ap@10"].mean(),
                "image_ndcg@10": image_subset["ndcg@10"].mean(),
                "cluster_p@10": cluster_subset["cluster_precision@10"].mean(),
                "cluster_image_recall@10": cluster_subset["image_recall_from_clusters@10"].mean(),
                "cluster_mrr": cluster_subset["cluster_mrr"].mean(),
            }
        )

    benchmark_rows = []
    for benchmark_type in sorted(image_df["benchmark_type"].unique()):
        image_subset = image_df[image_df["benchmark_type"] == benchmark_type]
        cluster_subset = cluster_df[cluster_df["benchmark_type"] == benchmark_type]
        benchmark_rows.append(
            {
                "query_type": benchmark_type,
                "family": query_family(benchmark_type),
                "queries": len(image_subset),
                "image_recall@5": image_subset["recall@5"].mean(),
                "image_recall@10": image_subset["recall@10"].mean(),
                "image_mrr": image_subset["mrr"].mean(),
                "image_ap@10": image_subset["ap@10"].mean(),
                "image_ndcg@10": image_subset["ndcg@10"].mean(),
                "cluster_p@10": cluster_subset["cluster_precision@10"].mean(),
                "cluster_image_recall@10": cluster_subset["image_recall_from_clusters@10"].mean(),
            }
        )

    md_lines = ["# Query Browser Report", ""]
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Query Browser Report</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;line-height:1.45;background:#faf9f6;color:#222;}",
        "h1,h2,h3{margin-top:1.2em;}",
        ".hero{background:#1f2937;color:white;border-radius:16px;padding:22px 26px;margin-bottom:18px;}",
        ".hero code{background:rgba(255,255,255,.15);color:white;}",
        ".section{margin-top:34px;padding-top:8px;border-top:3px solid #ddd;}",
        ".card{background:white;border:1px solid #ddd;border-radius:14px;padding:16px;margin:20px 0;box-shadow:0 1px 5px rgba(0,0,0,.04);}",
        ".tag{display:inline-block;border-radius:999px;padding:3px 10px;margin-right:6px;font-size:12px;background:#eef2ff;color:#3730a3;}",
        ".tag.general{background:#ecfdf5;color:#047857;}",
        ".tag.specific{background:#fff7ed;color:#c2410c;}",
        ".tag.public{background:#ecfdf5;color:#047857;}",
        ".tag.research{background:#eff6ff;color:#1d4ed8;}",
        ".tag.named{background:#fff7ed;color:#c2410c;}",
        ".metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px 16px;margin:12px 0;}",
        ".metric{background:#f7f7f7;padding:8px 10px;border-radius:8px;}",
        "table{border-collapse:collapse;width:100%;margin:10px 0 16px 0;}",
        "th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top;}",
        "th{background:#f0f0f0;}",
        "img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px;}",
        "code{background:#f4f4f4;padding:1px 4px;border-radius:4px;}",
        ".toc a{margin-right:12px;}",
        "</style></head><body>",
        "<div class='hero'>",
        "<h1>Search Evaluation Browser</h1>",
        "<p>This report evaluates TimeAtlas-style historical image retrieval with five query intentions: public visual concepts, public scene descriptions, research architectural evidence, research spatial evidence, and named place/landmark lookup.</p>",
        "<p>",
        f"Backend: <code>{escape_html(config.get('backend', ''))}</code> ",
        f"Index: <code>{escape_html(config.get('es_index', ''))}</code> ",
        f"Candidate K: <code>{escape_html(config.get('candidate_k', ''))}</code> ",
        f"Temperature: <code>{escape_html(config.get('temperature', ''))}</code> "
        f"Weights: <code>cluster={escape_html(config.get('cluster_weight', ''))}, "
        f"cls={escape_html(config.get('cls_weight', ''))}, "
        f"metadata={escape_html(config.get('metadata_weight', ''))}</code>",
        f" Dynamic metadata: <code>{escape_html(config.get('dynamic_metadata_weight', ''))}</code>",
        f" Image aggregation: <code>{escape_html(config.get('image_aggregation', ''))}, "
        f"k={escape_html(config.get('image_agg_top_k', ''))}, "
        f"alpha={escape_html(config.get('image_agg_alpha', ''))}</code>",
        "</p>",
        "</div>",
        "<div class='toc'><strong>Jump:</strong> <a href='#overview'>Overview</a><a href='#public'>Public queries</a><a href='#research'>Research queries</a><a href='#named'>Named queries</a><a href='#logic'>Benchmark logic</a></div>",
    ]

    def _table_html(df, cols, title):
        html_bits = [f"<h3>{escape_html(title)}</h3>"]
        if df.empty:
            html_bits.append("<p><em>None</em></p>")
            return "".join(html_bits)
        html_bits.append("<table><thead><tr>")
        for col in cols:
            html_bits.append(f"<th>{escape_html(col)}</th>")
        html_bits.append("</tr></thead><tbody>")
        for _, row in df.iterrows():
            html_bits.append("<tr>")
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    display = f"{value:.4f}" if not pd.isna(value) else ""
                else:
                    display = str(value)
                html_bits.append(f"<td>{escape_html(display)}</td>")
            html_bits.append("</tr>")
        html_bits.append("</tbody></table>")
        return "".join(html_bits)

    html_parts.append("<section id='overview' class='section'><h2>Evaluation overview</h2>")
    html_parts.append("<p>Results are ranked at image level, but each returned image is explained by all high-scoring regions in that image. Heatmaps use a shared query-score scale and show the top-scoring region percentage, rather than rendering one isolated cluster at a time.</p>")
    html_parts.append(
        _table_html(
            pd.DataFrame(overview_rows),
            ["family", "queries", "image_recall@5", "image_recall@10", "image_mrr", "image_ap@10", "image_ndcg@10", "cluster_p@10", "cluster_image_recall@10", "cluster_mrr"],
            "Macro metrics by query family",
        )
    )
    html_parts.append(
        _table_html(
            pd.DataFrame(benchmark_rows),
            ["query_type", "family", "queries", "image_recall@5", "image_recall@10", "image_mrr", "image_ap@10", "image_ndcg@10", "cluster_p@10", "cluster_image_recall@10"],
            "Macro metrics by benchmark type",
        )
    )
    html_parts.append("</section>")

    current_family = None

    for benchmark_type, query in query_keys:
        family = query_family(benchmark_type)
        if family != current_family:
            if current_family is not None:
                html_parts.append("</section>")
            current_family = family
            title = {
                "public": "Public-facing exploratory queries",
                "research": "Research-oriented evidence queries",
                "named": "Named place and landmark queries",
                "general": "General queries",
                "specific": "Specific place and landmark queries",
            }.get(family, family.title())
            html_parts.append(f"<section id='{family}' class='section'><h2>{title}</h2>")

        img_row = image_lookup[(benchmark_type, query)]
        clu_row = cluster_lookup[(benchmark_type, query)]

        gt_subset = gt_pairs_df[(gt_pairs_df["benchmark_type"] == benchmark_type) & (gt_pairs_df["query"] == query)].copy()
        image_hits_subset = image_hits_df[(image_hits_df["benchmark_type"] == benchmark_type) & (image_hits_df["query"] == query)].copy()
        cluster_hits_subset = cluster_hits_df[(cluster_hits_df["benchmark_type"] == benchmark_type) & (cluster_hits_df["query"] == query)].copy()

        image_hits_subset = image_hits_subset.sort_values("rank").head(10)
        cluster_hits_subset = cluster_hits_subset.sort_values("rank").head(10)
        gt_subset = gt_subset.head(12)

        query_slug = slugify_for_filename(query)
        vis_rel = None
        if visualizations_dir:
            for suffix in ["heatmap", "cluster_mode"]:
                vis_path = os.path.join(visualizations_dir, benchmark_type, f"{query_slug}_{suffix}.png")
                if os.path.exists(vis_path):
                    vis_rel = os.path.relpath(vis_path, report_dir).replace("\\", "/")
                    break

        md_lines.append(f"## {family} / {benchmark_type}: `{query}`")
        md_lines.append("")
        md_lines.append(f"- Support: `{int(img_row['support'])}`")
        md_lines.append(f"- Image metrics: `R@1={img_row['recall@1']:.4f}`, `R@5={img_row['recall@5']:.4f}`, `R@10={img_row['recall@10']:.4f}`, `MRR={img_row['mrr']:.4f}`, `AP@10={img_row['ap@10']:.4f}`, `nDCG@10={img_row['ndcg@10']:.4f}`")
        md_lines.append(f"- Weights: `region={img_row['cluster_weight']:.3f}`, `CLS={img_row['cls_weight']:.3f}`, `metadata={img_row['metadata_weight']:.3f}`")
        md_lines.append(f"- Cluster metrics: `P@10={clu_row['cluster_precision@10']:.4f}`, `P@20={clu_row['cluster_precision@20']:.4f}`, `ImageRecall@10={clu_row['image_recall_from_clusters@10']:.4f}`, `ClusterMRR={clu_row['cluster_mrr']:.4f}`")
        if vis_rel:
            md_lines.append(f"- Visualization: [{vis_rel}]({vis_rel})")
            md_lines.append("")
            md_lines.append(f"![{query}]({vis_rel})")
        md_lines.append("")

        html_parts.append(f"<div class='card'><h2><span class='tag {family}'>{family}</span><span class='tag'>{benchmark_type}</span><code>{escape_html(query)}</code></h2>")
        html_parts.append(f"<p><strong>Support:</strong> {int(img_row['support'])}</p>")
        html_parts.append("<div class='metrics'>")
        for label, value in [
            ("Recall@1", img_row["recall@1"]),
            ("Recall@5", img_row["recall@5"]),
            ("Recall@10", img_row["recall@10"]),
            ("MRR", img_row["mrr"]),
            ("AP@10", img_row["ap@10"]),
            ("nDCG@10", img_row["ndcg@10"]),
            ("Metadata weight", img_row["metadata_weight"]),
            ("Cluster P@10", clu_row["cluster_precision@10"]),
            ("Cluster P@20", clu_row["cluster_precision@20"]),
            ("ImageRecallFromClusters@10", clu_row["image_recall_from_clusters@10"]),
            ("Cluster MRR", clu_row["cluster_mrr"]),
        ]:
            html_parts.append(f"<div class='metric'><strong>{escape_html(label)}</strong><br>{float(value):.4f}</div>")
        html_parts.append("</div>")
        if vis_rel:
            html_parts.append(f"<h3>Search visualization</h3><img src='{escape_html(vis_rel)}' alt='{escape_html(query)}'>")

        html_parts.append(
            _table_html(
                gt_subset[["image_id", "final_city", "final_place", "landmarks_identified"]]
                if not gt_subset.empty else gt_subset,
                ["image_id", "final_city", "final_place", "landmarks_identified"],
                "Ground truth examples (first 12)",
            )
        )
        html_parts.append(
            _table_html(
                image_hits_subset[["rank", "image_id", "cluster_id", "score", "cluster_score", "cls_score", "metadata_score", "metadata_weight", "image_aggregation", "is_relevant", "final_city", "final_place"]]
                if not image_hits_subset.empty else image_hits_subset,
                ["rank", "image_id", "cluster_id", "score", "cluster_score", "cls_score", "metadata_score", "metadata_weight", "image_aggregation", "is_relevant", "final_city", "final_place"],
                "Top image-level hits (first 10)",
            )
        )
        html_parts.append(
            _table_html(
                cluster_hits_subset[["rank", "image_id", "cluster_id", "score", "cluster_score", "cls_score", "metadata_score", "metadata_weight", "is_relevant", "final_city", "final_place"]]
                if not cluster_hits_subset.empty else cluster_hits_subset,
                ["rank", "image_id", "cluster_id", "score", "cluster_score", "cls_score", "metadata_score", "metadata_weight", "is_relevant", "final_city", "final_place"],
                "Top cluster-level hits (first 10)",
            )
        )
        html_parts.append("</div>")

    if current_family is not None:
        html_parts.append("</section>")

    html_parts.append(
        "<section id='logic' class='section'><h2>Benchmark logic kept in this report</h2>"
        "<ul>"
        "<li><strong>Public visual queries</strong>: short user-style terms such as castle, church, bridge, river, boat, fountain, street, and tower.</li>"
        "<li><strong>Public scene queries</strong>: simple natural-language descriptions such as bridge over river, boats on water, and fountain in a square. Pseudo labels require multiple metadata concepts to match.</li>"
        "<li><strong>Research queries</strong>: architectural and spatial evidence queries such as church tower, dome, arched window, market square, harbor with boats, and riverfront with buildings.</li>"
        "<li><strong>Named queries</strong>: place and landmark queries generated from metadata fields. These are expected to benefit most from metadata fusion.</li>"
        "<li><strong>Image-level evaluation</strong>: each image is ranked by its best matching cluster; metrics include Recall@1/5/10, MRR, AP@10, and nDCG@10.</li>"
        "<li><strong>Cluster-level analysis</strong>: clusters are ranked directly; metrics include cluster precision and how many relevant images appear through top clusters.</li>"
        "<li><strong>Visual inspection</strong>: each result card displays the returned image with a heatmap over all high-scoring regions in that image. It does not split clusters into separate panels, so the overlay better matches the product experience.</li>"
        "</ul></section>"
    )
    html_parts.append("</body></html>")

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines))
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write("".join(html_parts))
    return md_path, html_path


def write_markdown_report(path, config, image_rows, cluster_rows):
    def macro(rows, key):
        values = [row[key] for row in rows if not pd.isna(row[key])]
        return float(np.mean(values)) if values else 0.0

    image_df = pd.DataFrame(image_rows)
    cluster_df = pd.DataFrame(cluster_rows)

    lines = []
    lines.append("# Metadata Search Evaluation")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append(f"- Backend: `{config['backend']}`")
    lines.append(f"- Index: `{config['es_index']}`")
    lines.append(f"- Candidate K: `{config['candidate_k']}`")
    lines.append(f"- Negative prompts: `{config['negative_text']}`")
    lines.append(f"- Temperature: `{config['temperature']}`")
    lines.append(f"- Cluster-centric weights: `cluster={config.get('cluster_weight')}`, `cls={config.get('cls_weight')}`, `metadata={config.get('metadata_weight')}`")
    lines.append(f"- Image aggregation: `{config.get('image_aggregation')}`, `k={config.get('image_agg_top_k')}`, `alpha={config.get('image_agg_alpha')}`")
    lines.append(f"- Query preset: `{config.get('query_preset')}`")
    lines.append(f"- Dynamic metadata weight: `{config.get('dynamic_metadata_weight')}`")
    if config.get("visualizations_dir"):
        lines.append(f"- Visualizations: `{config['visualizations_dir']}`")
    lines.append("")
    lines.append("## Image-Level Macro Metrics")
    lines.append("")
    for benchmark_type in sorted(image_df["benchmark_type"].unique()):
        subset = image_df[image_df["benchmark_type"] == benchmark_type]
        lines.append(f"### {benchmark_type}")
        lines.append("")
        lines.append(f"- Queries: `{len(subset)}`")
        lines.append(f"- Recall@1: `{subset['recall@1'].mean():.4f}`")
        lines.append(f"- Recall@5: `{subset['recall@5'].mean():.4f}`")
        lines.append(f"- Recall@10: `{subset['recall@10'].mean():.4f}`")
        lines.append(f"- MRR: `{subset['mrr'].mean():.4f}`")
        lines.append(f"- AP@10: `{subset['ap@10'].mean():.4f}`")
        lines.append(f"- nDCG@10: `{subset['ndcg@10'].mean():.4f}`")
        lines.append("")
    lines.append("## Cluster-Level Macro Metrics")
    lines.append("")
    for benchmark_type in sorted(cluster_df["benchmark_type"].unique()):
        subset = cluster_df[cluster_df["benchmark_type"] == benchmark_type]
        lines.append(f"### {benchmark_type}")
        lines.append("")
        lines.append(f"- Queries: `{len(subset)}`")
        lines.append(f"- Cluster Precision@10: `{subset['cluster_precision@10'].mean():.4f}`")
        lines.append(f"- Cluster Precision@20: `{subset['cluster_precision@20'].mean():.4f}`")
        lines.append(f"- Image Recall From Clusters@10: `{subset['image_recall_from_clusters@10'].mean():.4f}`")
        lines.append(f"- Image Recall From Clusters@20: `{subset['image_recall_from_clusters@20'].mean():.4f}`")
        lines.append(f"- Cluster MRR: `{subset['cluster_mrr'].mean():.4f}`")
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Evaluate image-level and cluster-level text search using metadata-derived weak labels.")
    parser.add_argument("--metadata_csv", default="images_metadata.csv")
    parser.add_argument("--es_host", default="http://localhost:9200")
    parser.add_argument("--es_timeout", type=float, default=60.0)
    parser.add_argument("--es_index", required=True)
    parser.add_argument("--backend", default="radseg", choices=["radseg", "tips", "talk2dino"])
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--model_version", default="c-radio_v4-h")
    parser.add_argument("--lang_model", default="siglip2-g")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vector_field", default="vector")
    parser.add_argument("--negative_text", default="background, sky, clouds, text, border, trees, road, people")
    parser.add_argument("--candidate_k", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--cluster_weight", type=float, default=1.0)
    parser.add_argument("--cls_weight", type=float, default=0.2)
    parser.add_argument("--metadata_weight", type=float, default=0.3)
    parser.add_argument("--dynamic_metadata_weight", action="store_true")
    parser.add_argument("--image_aggregation", choices=["max", "topk_mean", "max_topk_mean"], default="max_topk_mean")
    parser.add_argument("--image_agg_top_k", type=int, default=3)
    parser.add_argument("--image_agg_alpha", type=float, default=0.7)
    parser.add_argument("--image_top_k", type=int, default=20)
    parser.add_argument("--cluster_top_k", type=int, default=50)
    parser.add_argument("--min_city_count", type=int, default=10)
    parser.add_argument("--min_place_count", type=int, default=5)
    parser.add_argument("--min_landmark_count", type=int, default=3)
    parser.add_argument("--top_n_city", type=int, default=0)
    parser.add_argument("--top_n_place", type=int, default=6)
    parser.add_argument("--top_n_landmark", type=int, default=20)
    parser.add_argument("--query_preset", choices=["legacy", "europeana"], default="europeana")
    parser.add_argument("--report_dir", default=None)
    parser.add_argument("--redis_url", default="redis://localhost:6379/0")
    parser.add_argument("--redis_key_prefix", default="fm")
    parser.add_argument("--image_root", default="images")
    parser.add_argument("--visualize_queries", choices=["all", "public", "research", "named", "concept", "structured", "none"], default="all")
    parser.add_argument("--visualize_result_mode", choices=["image", "cluster"], default="image")
    parser.add_argument("--visualize_top_k", type=int, default=6)
    parser.add_argument("--heatmap_top_percent", type=float, default=35.0)
    parser.add_argument("--heatmap_min_score", type=float, default=None)
    parser.add_argument("--offline", action="store_true", help="Use local Hugging Face cache only")
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if args.report_dir is None:
        args.report_dir = os.path.join("scratch", f"eval_{args.backend}_{args.es_index}")
    os.makedirs(args.report_dir, exist_ok=True)

    es = Elasticsearch(args.es_host, request_timeout=args.es_timeout)
    if not es.ping():
        raise SystemExit(f"Could not connect to Elasticsearch at {args.es_host}")

    indexed_image_ids = get_indexed_image_ids(es, args.es_index)
    print(f"Indexed images in {args.es_index}: {len(indexed_image_ids)}")

    metadata_df = prepare_metadata(args.metadata_csv, indexed_image_ids)
    print(f"Metadata rows aligned to index: {len(metadata_df)}")

    name_lexicon = build_name_lexicon(metadata_df)
    if args.query_preset == "europeana":
        structured_queries = build_europeana_queries(
            metadata_df,
            min_place_count=args.min_place_count,
            min_landmark_count=args.min_landmark_count,
            top_n_place=args.top_n_place,
            top_n_landmark=args.top_n_landmark,
        )
        concept_queries = []
    else:
        structured_queries = build_structured_queries(
            metadata_df,
            min_city_count=args.min_city_count,
            min_place_count=args.min_place_count,
            min_landmark_count=args.min_landmark_count,
            top_n_city=args.top_n_city,
            top_n_place=args.top_n_place,
            top_n_landmark=args.top_n_landmark,
        )
        concept_queries = build_concept_queries(metadata_df)
    all_queries = [query for query in structured_queries + concept_queries if query["support"] > 0]

    print(f"Total benchmark queries: {len(all_queries)}")

    gt_json_path, gt_csv_path = export_ground_truth(args.report_dir, all_queries, metadata_df)

    scorer = SearchScorer(
        backend_name=args.backend,
        es_host=args.es_host,
        es_index=args.es_index,
        device=args.device,
        es_timeout=args.es_timeout,
        model_id=args.model_id,
        model_version=args.model_version,
        lang_model=args.lang_model,
        vector_field=args.vector_field,
        candidate_k=args.candidate_k,
        negative_text=args.negative_text,
        temperature=args.temperature,
        cluster_weight=args.cluster_weight,
        cls_weight=args.cls_weight,
        metadata_weight=args.metadata_weight,
        image_aggregation=args.image_aggregation,
        image_agg_top_k=args.image_agg_top_k,
        image_agg_alpha=args.image_agg_alpha,
    )

    visualizations_dir = None
    visualizer = None
    if args.visualize_queries != "none":
        visualizations_dir = os.path.join(args.report_dir, "visualizations")
        os.makedirs(visualizations_dir, exist_ok=True)
        visualizer = TextSearchVisualizer(
            backend_name=args.backend,
            es_host=args.es_host,
            es_index=args.es_index,
            redis_url=args.redis_url,
            redis_key_prefix=args.redis_key_prefix,
            image_root=args.image_root,
            device=args.device,
            vector_field=args.vector_field,
            model_id=args.model_id,
            model_version=args.model_version,
            lang_model=args.lang_model,
            cluster_weight=args.cluster_weight,
            cls_weight=args.cls_weight,
            metadata_weight=args.metadata_weight,
        )
        visualizer.es = Elasticsearch(args.es_host, request_timeout=args.es_timeout)

    image_rows = []
    cluster_rows = []
    cluster_hit_rows = []
    image_hit_rows = []
    timing_rows = []
    metadata_lookup = metadata_df.set_index("image_filename")

    for query_spec in tqdm(all_queries, desc="Evaluating queries"):
        query_weights = compute_query_weights(query_spec, args, name_lexicon)
        image_results, image_timing = scorer.search(
            query_spec["query"],
            result_mode="image",
            top_k=args.image_top_k,
            weights=query_weights,
            return_timing=True,
        )
        cluster_results, cluster_timing = scorer.search(
            query_spec["query"],
            result_mode="cluster",
            top_k=args.cluster_top_k,
            weights=query_weights,
            return_timing=True,
        )

        image_metrics = evaluate_image_level(query_spec, image_results)
        cluster_metrics = evaluate_cluster_level(query_spec, cluster_results)

        image_rows.append(
            {
                "benchmark_type": query_spec["benchmark_type"],
                "query_family": query_family(query_spec["benchmark_type"]),
                "query": query_spec["query"],
                "label": query_spec["label"],
                "support": query_spec["support"],
                "query_group": query_group_label(query_spec["benchmark_type"]),
                **query_weights,
                **image_metrics,
            }
        )
        cluster_rows.append(
            {
                "benchmark_type": query_spec["benchmark_type"],
                "query_family": query_family(query_spec["benchmark_type"]),
                "query": query_spec["query"],
                "label": query_spec["label"],
                "support": query_spec["support"],
                "query_group": query_group_label(query_spec["benchmark_type"]),
                **query_weights,
                **cluster_metrics,
            }
        )

        visualization_timing = {
            "visual_full_cluster_scoring_s": 0.0,
            "redis_fetch_s": 0.0,
            "heatmap_reconstruction_s": 0.0,
            "figure_render_save_s": 0.0,
            "visualization_total_s": 0.0,
        }
        if visualizer is not None:
            should_render = (
                args.visualize_queries == "all"
                or (args.visualize_queries == query_family(query_spec["benchmark_type"]))
                or (args.visualize_queries == "concept" and query_spec["benchmark_type"] == "concept")
                or (args.visualize_queries == "structured" and query_spec["benchmark_type"] != "concept")
            )
            if should_render:
                visualizer.cluster_weight = query_weights["cluster_weight"]
                visualizer.cls_weight = query_weights["cls_weight"]
                visualizer.metadata_weight = query_weights["metadata_weight"]
                query_slug = slugify_for_filename(query_spec["query"])
                mode_suffix = "cluster_mode" if args.visualize_result_mode == "cluster" else "heatmap"
                query_dir = os.path.join(visualizations_dir, query_spec["benchmark_type"])
                os.makedirs(query_dir, exist_ok=True)
                output_path = os.path.join(query_dir, f"{query_slug}_{mode_suffix}.png")
                render_results = cluster_results if args.visualize_result_mode == "cluster" else image_results
                visualization_timing = visualizer.visualize_results(
                    render_results[: args.visualize_top_k],
                    query_text=query_spec["query"],
                    negative_prompts=visualizer.normalize_negative_prompts(args.negative_text),
                    output_path=output_path,
                    result_mode=args.visualize_result_mode,
                    temperature=args.temperature,
                    heatmap_top_percent=args.heatmap_top_percent,
                    heatmap_min_score=args.heatmap_min_score,
                    return_timing=True,
                )
        timing_rows.append(
            {
                "benchmark_type": query_spec["benchmark_type"],
                "query_family": query_family(query_spec["benchmark_type"]),
                "query_group": query_group_label(query_spec["benchmark_type"]),
                "query": query_spec["query"],
                "support": query_spec["support"],
                **image_timing,
                "cluster_eval_total_s": cluster_timing["total_latency_s"],
                **visualization_timing,
                "end_to_end_with_visualization_s": image_timing["total_latency_s"]
                + visualization_timing["visualization_total_s"],
            }
        )

        relevant = query_spec["relevant_images"]
        for rank, item in enumerate(image_results, start=1):
            image_id = item["image_id"]
            if image_id in metadata_lookup.index:
                row = metadata_lookup.loc[image_id]
                final_city = row["final_city_clean"]
                final_place = row["final_place_clean"]
                landmarks_identified = row["landmarks_clean"]
                description = row["description_clean"][:240]
            else:
                final_city = ""
                final_place = ""
                landmarks_identified = ""
                description = ""
            image_hit_rows.append(
                {
                    "benchmark_type": query_spec["benchmark_type"],
                    "query_family": query_family(query_spec["benchmark_type"]),
                    "query": query_spec["query"],
                    "support": query_spec["support"],
                    "rank": rank,
                    "image_id": image_id,
                    "cluster_id": item["cluster_id"],
                    "score": item["score"],
                    "cluster_score": item.get("cluster_score", 0.0),
                    "cls_score": item.get("cls_score", 0.0),
                    "metadata_score": item.get("metadata_score", 0.0),
                    "cluster_weight": query_weights["cluster_weight"],
                    "cls_weight": query_weights["cls_weight"],
                    "metadata_weight": query_weights["metadata_weight"],
                    "image_aggregation": item.get("image_aggregation", args.image_aggregation),
                    "image_agg_top_k": item.get("image_agg_top_k", args.image_agg_top_k),
                    "image_agg_alpha": item.get("image_agg_alpha", args.image_agg_alpha),
                    "is_relevant": int(image_id in relevant),
                    "final_city": final_city,
                    "final_place": final_place,
                    "landmarks_identified": landmarks_identified,
                    "description": description,
                }
            )
        for rank, item in enumerate(cluster_results, start=1):
            image_id = item["image_id"]
            if image_id in metadata_lookup.index:
                row = metadata_lookup.loc[image_id]
                final_city = row["final_city_clean"]
                final_place = row["final_place_clean"]
                landmarks_identified = row["landmarks_clean"]
                description = row["description_clean"][:240]
            else:
                final_city = ""
                final_place = ""
                landmarks_identified = ""
                description = ""
            cluster_hit_rows.append(
                {
                    "benchmark_type": query_spec["benchmark_type"],
                    "query_family": query_family(query_spec["benchmark_type"]),
                    "query": query_spec["query"],
                    "support": query_spec["support"],
                    "rank": rank,
                    "image_id": image_id,
                    "cluster_id": item["cluster_id"],
                    "score": item["score"],
                    "cluster_score": item.get("cluster_score", 0.0),
                    "cls_score": item.get("cls_score", 0.0),
                    "metadata_score": item.get("metadata_score", 0.0),
                    "cluster_weight": query_weights["cluster_weight"],
                    "cls_weight": query_weights["cls_weight"],
                    "metadata_weight": query_weights["metadata_weight"],
                    "is_relevant": int(image_id in relevant),
                    "final_city": final_city,
                    "final_place": final_place,
                    "landmarks_identified": landmarks_identified,
                    "description": description,
                }
            )

    image_df = pd.DataFrame(image_rows).sort_values(["query_family", "benchmark_type", "query"])
    cluster_df = pd.DataFrame(cluster_rows).sort_values(["query_family", "benchmark_type", "query"])
    image_hits_df = pd.DataFrame(image_hit_rows).sort_values(["query_family", "query", "rank"])
    cluster_hits_df = pd.DataFrame(cluster_hit_rows).sort_values(["query_family", "query", "rank"])
    timing_df = pd.DataFrame(timing_rows).sort_values(["query_family", "benchmark_type", "query"])
    gt_pairs_df = pd.read_csv(gt_csv_path, encoding="utf-8")

    image_csv = os.path.join(args.report_dir, "image_level_metrics.csv")
    cluster_csv = os.path.join(args.report_dir, "cluster_level_metrics.csv")
    image_hits_csv = os.path.join(args.report_dir, "image_level_top_hits.csv")
    cluster_hits_csv = os.path.join(args.report_dir, "cluster_level_top_hits.csv")
    timing_csv = os.path.join(args.report_dir, "query_timing.csv")
    timing_summary_csv = os.path.join(args.report_dir, "query_timing_summary.csv")
    summary_json = os.path.join(args.report_dir, "summary.json")
    summary_md = os.path.join(args.report_dir, "summary.md")

    image_df.to_csv(image_csv, index=False, encoding="utf-8")
    cluster_df.to_csv(cluster_csv, index=False, encoding="utf-8")
    image_hits_df.to_csv(image_hits_csv, index=False, encoding="utf-8")
    cluster_hits_df.to_csv(cluster_hits_csv, index=False, encoding="utf-8")
    timing_df.to_csv(timing_csv, index=False, encoding="utf-8")

    timing_columns = [
        "query_encoding_s",
        "es_knn_search_s",
        "negative_rerank_s",
        "cls_metadata_scoring_s",
        "image_aggregation_s",
        "total_latency_s",
        "visual_full_cluster_scoring_s",
        "redis_fetch_s",
        "heatmap_reconstruction_s",
        "figure_render_save_s",
        "visualization_total_s",
        "end_to_end_with_visualization_s",
    ]
    timing_summary_rows = []
    for group_name, group_df in [("all", timing_df)] + list(timing_df.groupby("query_family")):
        row = {"query_family": group_name, "queries": len(group_df)}
        for column in timing_columns:
            if column not in group_df:
                continue
            row[f"{column}_mean"] = float(group_df[column].mean())
            row[f"{column}_p95"] = float(group_df[column].quantile(0.95))
        timing_summary_rows.append(row)
    timing_summary_df = pd.DataFrame(timing_summary_rows)
    timing_summary_df.to_csv(timing_summary_csv, index=False, encoding="utf-8")

    config = {
        "backend": args.backend,
        "es_host": args.es_host,
        "es_timeout": args.es_timeout,
        "es_index": args.es_index,
        "candidate_k": args.candidate_k,
        "image_top_k": args.image_top_k,
        "cluster_top_k": args.cluster_top_k,
        "negative_text": args.negative_text,
        "temperature": args.temperature,
        "cluster_weight": args.cluster_weight,
        "cls_weight": args.cls_weight,
        "metadata_weight": args.metadata_weight,
        "dynamic_metadata_weight": args.dynamic_metadata_weight,
        "image_aggregation": args.image_aggregation,
        "image_agg_top_k": args.image_agg_top_k,
        "image_agg_alpha": args.image_agg_alpha,
        "heatmap_top_percent": args.heatmap_top_percent,
        "heatmap_min_score": args.heatmap_min_score,
        "query_preset": args.query_preset,
        "visualizations_dir": visualizations_dir,
        "ground_truth_json": gt_json_path,
        "ground_truth_csv": gt_csv_path,
        "indexed_images": len(indexed_image_ids),
        "aligned_metadata_rows": len(metadata_df),
        "structured_queries": len(structured_queries),
        "concept_queries": len(concept_queries),
        "total_queries": len(all_queries),
        "name_lexicon_size": len(name_lexicon),
    }

    summary = {
        "config": config,
        "image_level_family_macro": image_df.groupby("query_family")[["recall@1", "recall@5", "recall@10", "mrr", "ap@10", "ndcg@10"]].mean().round(4).to_dict(orient="index"),
        "cluster_level_family_macro": cluster_df.groupby("query_family")[["cluster_precision@10", "cluster_precision@20", "image_recall_from_clusters@10", "image_recall_from_clusters@20", "cluster_mrr"]].mean().round(4).to_dict(orient="index"),
        "image_level_macro": image_df.groupby("benchmark_type")[["recall@1", "recall@5", "recall@10", "mrr", "ap@10", "ndcg@10"]].mean().round(4).to_dict(orient="index"),
        "cluster_level_macro": cluster_df.groupby("benchmark_type")[["cluster_precision@10", "cluster_precision@20", "image_recall_from_clusters@10", "image_recall_from_clusters@20", "cluster_mrr"]].mean().round(4).to_dict(orient="index"),
        "timing_mean": timing_df[timing_columns].mean().round(4).to_dict(),
        "timing_p95": timing_df[timing_columns].quantile(0.95).round(4).to_dict(),
    }

    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_markdown_report(summary_md, config, image_rows, cluster_rows)
    query_report_md, query_report_html = write_query_browser_report(
        args.report_dir,
        image_df,
        cluster_df,
        gt_pairs_df,
        image_hits_df,
        cluster_hits_df,
        visualizations_dir,
        config=config,
    )

    print(f"Saved image-level metrics to {image_csv}")
    print(f"Saved cluster-level metrics to {cluster_csv}")
    print(f"Saved image-level hit analysis to {image_hits_csv}")
    print(f"Saved cluster hit analysis to {cluster_hits_csv}")
    print(f"Saved query timing to {timing_csv}")
    print(f"Saved query timing summary to {timing_summary_csv}")
    print(f"Saved ground truth JSON to {gt_json_path}")
    print(f"Saved ground truth CSV to {gt_csv_path}")
    print(f"Saved summary to {summary_json}")
    print(f"Saved markdown report to {summary_md}")
    print(f"Saved query browser markdown to {query_report_md}")
    print(f"Saved query browser HTML to {query_report_html}")


if __name__ == "__main__":
    main()
