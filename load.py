import json
import csv
import os
from re import template
from numpy import double
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
word_list = tokenizer.get_vocab()

CF_TOKENS = ["dasjhasjkdhjskdhds", "N/A", "BLANK"]
def loadHeShe(tokenizer=tokenizer):
    return ['he', 'she'], [tokenizer.encode('he', add_special_tokens=False), tokenizer.encode('she', add_special_tokens=False)]

def loadManWoman(tokenizer=tokenizer):
    return ['man', 'woman'], [tokenizer.encode('man', add_special_tokens=False), tokenizer.encode('woman', add_special_tokens=False)]

def loadGenderedWords(tokenizer=tokenizer, gendered_words_dir=os.path.join(current_dir, 'data/gender.json')):
    f_ids = []
    m_ids = []
    f_words = []
    m_words = []
    with open(gendered_words_dir, 'r', encoding='utf-8') as rf:
        data = json.load(rf) 
        for dict in data:
            if dict['gender'] == 'f':
                try:
                    buf_id = tokenizer.encode(dict['word'], add_special_tokens=False)
                    if buf_id not in f_ids:
                        f_words.append(dict['word'])
                        f_ids.append(buf_id)
                    buf_id = tokenizer.encode(dict['gender_map']['m']['word'], add_special_token=False)
                    if buf_id not in m_ids:
                        m_words.append(dict['gender_map']['m']['word'])
                        m_ids.append(buf_id)
                except:
                    pass
            elif dict['gender'] == 'm':
                try:
                    buf_id = tokenizer.encode(dict['word'], add_special_tokens=False)
                    if buf_id not in m_ids:
                        m_words.append(dict['word'])
                        m_ids.append(buf_id)
                    buf_id = tokenizer.encode(dict['gender_map']['f']['word'], add_special_tokens=False)
                    if buf_id not in f_ids:
                        f_words.append(dict['gender_map']['f']['word'])
                        f_ids.append(buf_id)
                except:
                    pass
    return m_words, m_ids, f_words, f_ids

def loadTempletes(template_path, threshold=0.95):
    rf = open(template_path, 'r')
    reader = csv.reader(rf)
    reader = [line for line in reader]
    rf.close()
    _templates = []
    for line in reader[1:]:
        _templates.append((line[0].strip(), double(line[1])))
    _templates.sort(key=lambda item:item[1], reverse=True)
    _proportions = [item[1] for item in _templates]
    _total = sum(_proportions)
    _threshold = _total * threshold

    templates = []
    proportions = []
    _sum = 0
    for template in _templates:
        _t = template[0]
        _p = template[1]
        _sum += _p
        if _sum >= _threshold:
            break
        templates.append(_t)
        proportions.append(_p)
    return templates, proportions


def loadOccupationM(tokenizer=tokenizer, occupations_dir=os.path.join(current_dir, 'data/occupation.csv')):
    with open(occupations_dir, 'r') as rf:
        occupation_reader = csv.reader(rf)
        occupations = []
        occupation_ids = []
        for occupation in occupation_reader:
            if occupation[1] == 'RES' and occupation[2] == 'RES' and occupation[3] == 'RES':
                try:
                    buf_id = tokenizer.encode(occupation[0], add_special_tokens=False)
                    occupations.append(occupation[0])
                    occupation_ids.append(buf_id)
                except:
                    pass
    return occupations, occupation_ids

def loadOccupation(tokenizer=tokenizer, occupations_dir=os.path.join(current_dir, 'data/occ.txt')):
    with open(occupations_dir, 'r') as rf:
        occupation_reader = rf.readlines()
        occupations = []
        occupation_ids = []
        for occupation in occupation_reader:
            occupation = occupation.strip()
            try:
                buf_id = tokenizer.encode(occupation, add_special_tokens=False)
                occupations.append(occupation)
                occupation_ids.append(buf_id)
            except:
                pass
        return occupations, occupation_ids

def loadReligion(tokenizer=tokenizer, religions_dir=os.path.join(current_dir, 'data/religion.json')):
    with open(religions_dir, 'r') as rf:
        religion_reader = json.load(rf)
        religions = []
        religion_ids = []
        for religion in religion_reader:
            try:
                buf_id = tokenizer.encode(religion, add_special_tokens=False)
                religions.append(religion)
                religion_ids.append(buf_id)
            except:
                pass
    return religions, religion_ids

def loadRace(tokenizer=tokenizer, races_dir=os.path.join(current_dir, 'data/race.json')):
    with open(races_dir, 'r') as rf:
        race_reader = json.load(rf)
        races = []
        race_ids = []
        for race in race_reader:
            try:
                buf_id = tokenizer.encode(race, add_special_tokens=False)
                races.append(race)
                race_ids.append(buf_id)
            except:
                pass
    return races, race_ids

def loadSkinColor(tokenizer=tokenizer, skin_colors_dir=os.path.join(current_dir, 'data/skin_color.json')):
    with open(skin_colors_dir, 'r') as rf:
        skin_color_reader = json.load(rf)
        skin_colors = []
        skin_color_ids = []
        for skin_color in skin_color_reader:
            try:
                buf_id = tokenizer.encode(skin_color, add_special_tokens=False)
                skin_colors.append(skin_color)
                skin_color_ids.append(buf_id)
            except:
                pass
    return skin_colors, skin_color_ids

def loadFamilyRole(tokenizer=tokenizer, family_roles_dir=os.path.join(current_dir, 'data/family_role.json')):
    with open(family_roles_dir, 'r') as rf:
        family_role_reader = json.load(rf)
        family_roles = []
        family_role_ids = []
        for family_role in family_role_reader:
            try:
                buf_id = tokenizer.encode(family_role, add_special_tokens=False)
                family_roles.append(family_role)
                family_role_ids.append(buf_id)
            except:
                pass
    return family_roles, family_role_ids

def loadFamilyType(tokenizer=tokenizer, family_types_dir=os.path.join(current_dir, 'data/family_type.json')):
    with open(family_types_dir, 'r') as rf:
        family_type_reader = json.load(rf)
        family_types = []
        family_type_ids = []
        for family_type in family_type_reader:
            try:
                buf_id = tokenizer.encode(family_type, add_special_tokens=False)
                family_types.append(family_type)
                family_type_ids.append(buf_id)
            except:
                pass
    return family_types, family_type_ids

def loadDisease(tokenizer=tokenizer, diseases_dir=os.path.join(current_dir, 'data/disease.json')):
    with open(diseases_dir, 'r') as rf:
        disease_reader = json.load(rf)
        diseases = []
        disease_ids = []
        for disease in disease_reader:
            try:
                buf_id = tokenizer.encode(disease, add_special_tokens=False)
                diseases.append(disease)
                disease_ids.append(buf_id)
            except:
                pass
    return diseases, disease_ids

def loadBodyShape(tokenizer=tokenizer, body_shapes_dir=os.path.join(current_dir, 'data/body_shape.json')):
    with open(body_shapes_dir, 'r') as rf:
        body_shape_reader = json.load(rf)
        body_shapes = []
        body_shape_ids = []
        for body_shape in body_shape_reader:
            try:
                buf_id = tokenizer.encode(body_shape, add_special_tokens=False)
                body_shapes.append(body_shape)
                body_shape_ids.append(buf_id)
            except:
                pass
    return body_shapes, body_shape_ids

def loadHairstyle(tokenizer=tokenizer, hairstyles_dir=os.path.join(current_dir, 'data/hairstyle.json')):
    with open(hairstyles_dir, 'r') as rf:
        hairstyle_reader = json.load(rf)
        hairstyles = []
        hairstyle_ids = []
        for hairstyle in hairstyle_reader:
            try:
                buf_id = tokenizer.encode(hairstyle, add_special_tokens=False)
                hairstyles.append(hairstyle)
                hairstyle_ids.append(buf_id)
            except:
                pass
    return hairstyles, hairstyle_ids

def loadAppearanceDefect(tokenizer=tokenizer, appearance_defects_dir=os.path.join(current_dir, 'data/appearance_defect.json')):
    with open(appearance_defects_dir, 'r') as rf:
        appearance_defect_reader = json.load(rf)
        appearance_defects = []
        appearance_defect_ids = []
        for appearance_defect in appearance_defect_reader:
            try:
                buf_id = tokenizer.encode(appearance_defect, add_special_tokens=False)
                appearance_defects.append(appearance_defect)
                appearance_defect_ids.append(buf_id)
            except:
                pass
    return appearance_defects, appearance_defect_ids

def loadPoliticalStand(tokenizer=tokenizer, political_stands_dir=os.path.join(current_dir, 'data/political_stand.json')):
    with open(political_stands_dir, 'r') as rf:
        political_stand_reader = json.load(rf)
        political_stands = []
        political_stand_ids = []
        for political_stand in political_stand_reader:
            try:
                buf_id = tokenizer.encode(political_stand, add_special_tokens=False)
                political_stands.append(political_stand)
                political_stand_ids.append(buf_id)
            except:
                pass
    return political_stands, political_stand_ids

def loadAge(tokenizer=tokenizer, ages_dir=os.path.join(current_dir, 'data/age.json')):
    with open(ages_dir, 'r') as rf:
        age_reader = json.load(rf)
        ages = []
        age_ids = []
        for age in age_reader:
            try:
                buf_id = tokenizer.encode(age, add_special_tokens=False)
                ages.append(age)
                age_ids.append(buf_id)
            except:
                pass
    return ages, age_ids

def loadDisability(tokenizer=tokenizer, disablities_dir=os.path.join(current_dir, 'data/disability.json')):
    with open(disablities_dir, 'r') as rf:
        disability_reader = json.load(rf)
        disabilities = []
        disability_ids = []
        for disability in disability_reader:
            try:
                buf_id = tokenizer.encode(disability, add_special_tokens=False)
                disabilities.append(disability)
                disability_ids.append(buf_id)
            except:
                pass
    return disabilities, disability_ids

def loadGenderIdentity(tokenizer=tokenizer, gender_identities_dir=os.path.join(current_dir, 'data/gender_identity.json')):
    with open(gender_identities_dir, 'r') as rf:
        gender_identity_reader = json.load(rf)
        gender_identities = []
        gender_identity_ids = []
        for gender_identity in gender_identity_reader:
            try:
                buf_id = tokenizer.encode(gender_identity, add_special_tokens=False)
                gender_identities.append(gender_identity)
                gender_identity_ids.append(buf_id)
            except:
                pass
    return gender_identities, gender_identity_ids

def loadPregnancy(tokenizer=tokenizer, pregnancys_dir=os.path.join(current_dir, 'data/pregnancy.json')):
    with open(pregnancys_dir, 'r') as rf:
        pregnancy_reader = json.load(rf)
        pregnancys = []
        pregnancy_ids = []
        for pregnancy in pregnancy_reader:
            try:
                buf_id = tokenizer.encode(pregnancy, add_special_tokens=False)
                pregnancys.append(pregnancy)
                pregnancy_ids.append(buf_id)
            except:
                pass
    return pregnancys, pregnancy_ids

def loadMentalDisorder(tokenizer=tokenizer, mental_disorders_dir=os.path.join(current_dir, 'data/mental_disorder.json')):
    with open(mental_disorders_dir, 'r') as rf:
        mental_disorder_reader = json.load(rf)
        mental_disorders = []
        mental_disorder_ids = []
        for mental_disorder in mental_disorder_reader:
            try:
                buf_id = tokenizer.encode(mental_disorder, add_special_tokens=False)
                mental_disorders.append(mental_disorder)
                mental_disorder_ids.append(buf_id)
            except:
                pass
    return mental_disorders, mental_disorder_ids

def loadCountry(tokenizer=tokenizer, nations_dir=os.path.join(current_dir, 'data/country.json')):
    with open(nations_dir, 'r') as rf:
        nation_reader = json.load(rf)
        nations = []
        nation_ids = []
        for nation in nation_reader:
            try:
                buf_id = tokenizer.encode(nation['nationality'], add_special_tokens=False)
                nations.append(nation['nationality'])
                nation_ids.append(buf_id)
            except:
                pass
    return nations, nation_ids
 
def loadUniversity(tokenizer=tokenizer, universities_dir=os.path.join(current_dir, 'data/university.csv')):
    with open(universities_dir, 'r') as rf:
        universitiy_reader = csv.reader(rf)
        universities = []
        universitiy_ids = []
        for university in universitiy_reader:
            university = university[1] + ',' + university[0]
            try:
                buf_id = tokenizer.encode(university, add_special_tokens=False)
                universities.append(university)
                universitiy_ids.append(buf_id)
            except:
                pass
    return universities, universitiy_ids

def loadProvince(tokenizer=tokenizer, provinces_dir=os.path.join(current_dir, 'data/province.json')):
    with open(provinces_dir, 'r') as rf:
        province_reader = json.load(rf)
        provinces = []
        province_ids = []
        for item in province_reader:
            if 'english' in item.keys():
                province = item['english']
            else:
                province = item['name']
            province = province + ',' + item['country']
            try:
                buf_id = tokenizer.encode(province, add_special_tokens=False)
                provinces.append(province)
                province_ids.append(buf_id)
            except:
                pass
    return provinces, province_ids

def loadGenderedWords_strict(tokenizer=tokenizer, gendered_words_dir=os.path.join(current_dir, 'data/gender.json')):
    f_ids = []
    m_ids = []
    f_words = []
    m_words = []
    with open(gendered_words_dir, 'r', encoding='utf-8') as rf:
        data = json.load(rf) 
        for dict in data:
            if dict['gender'] == 'f':
                try:
                    buf_id = word_list[dict['word']]
                    if buf_id not in f_ids:
                        f_words.append(dict['word'])
                        f_ids.append(buf_id)
                except:
                    pass
            elif dict['gender'] == 'm':
                try:
                    buf_id = word_list[dict['word']]
                    if buf_id not in m_ids:
                        m_words.append(dict['word'])
                        m_ids.append(buf_id)
                except:
                    pass
    return m_words, m_ids, f_words, f_ids

def loadSexualOrientation(tokenizer=tokenizer, sexual_orientations_dir=os.path.join(current_dir, 'data/sexual_orientation.json')):
    with open(sexual_orientations_dir, 'r') as rf:
        sexual_orientation_reader = json.load(rf)
        sexual_orientations = []
        sexual_orientation_ids = []
        for sexual_orientation in sexual_orientation_reader:
            try:
                buf_id = tokenizer.encode(sexual_orientation, add_special_tokens=False)
                sexual_orientations.append(sexual_orientation)
                sexual_orientation_ids.append(buf_id)
            except:
                pass
    return sexual_orientations, sexual_orientation_ids

def loadNationality(tokenizer=tokenizer, nationalities_dir=os.path.join(current_dir, 'data/nationality.json')):
    with open(nationalities_dir, 'r') as rf:
        nationality_reader = json.load(rf)
        nationalities = []
        nationality_ids = []
        for nationality in nationality_reader:
            try:
                buf_id = tokenizer.encode(nationality, add_special_tokens=False)
                nationalities.append(nationality)
                nationality_ids.append(buf_id)
            except:
                pass
    return nationalities, nationality_ids

LOAD_MAP = { 
            'heshe' : loadHeShe,
            'manwoman' : loadManWoman,
            'gendered word' : loadGenderedWords,
            'occupation' : loadOccupation, 
            'religion' : loadReligion, #
            'race' : loadRace, #
            'age'  : loadAge, #
            'skin color' : loadSkinColor,
            'family role' : loadFamilyRole,
            'family type' : loadFamilyType,
            'disease' : loadDisease,
            'body shape' : loadBodyShape,
            'hairstyle' : loadHairstyle,
            'appearance defect' : loadAppearanceDefect,
            'political stand' : loadPoliticalStand,
            'disability' : loadDisability, #
            'gender identity' : loadGenderIdentity,
            'pregnancy' : loadPregnancy,
            'mental disorder' : loadMentalDisorder,
            'country' : loadCountry,
            'province' : loadProvince, 
            'university' : loadUniversity,
            'sexual orientation' : loadSexualOrientation,
            'nationality' : loadNationality}