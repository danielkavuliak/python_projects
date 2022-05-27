import pandas as pd
from pymongo import MongoClient
import requests
from quartile import Quartile
import psycopg2


def create_tables(cursor, connection):
    sql = '''CREATE TABLE kraj_okres(
    id SERIAL PRIMARY KEY,
    datum VARCHAR(50),
    kraj_nuts_kod VARCHAR(50),
    okres_lau_kod VARCHAR(50),
    kumulativni_pocet_nakazenych INTEGER,
    kumulativni_pocet_vylecenych INTEGER,
    kumulativni_pocet_umrti INTEGER,
    x FLOAT,
    y FLOAT
    );'''

    cursor.execute(sql)
    connection.commit()

    sql = '''CREATE TABLE osoby(
    id SERIAL PRIMARY KEY,
    datum VARCHAR(50),
    vek INTEGER,
    pohlavi VARCHAR(50),
    kraj_nuts_kod VARCHAR(50),
    okres_lau_kod VARCHAR(50)
    );'''

    cursor.execute(sql)
    connection.commit()

    sql = '''CREATE TABLE umrti(
    id SERIAL PRIMARY KEY,
    datum VARCHAR(50),
    vek INTEGER,
    pohlavi VARCHAR(50),
    kraj_nuts_kod VARCHAR(50),
    okres_lau_kod VARCHAR(50)
    );'''

    cursor.execute(sql)
    connection.commit()

    sql = '''CREATE TABLE vyleceni(
    id SERIAL PRIMARY KEY,
    datum VARCHAR(50),
    vek INTEGER,
    pohlavi VARCHAR(50),
    kraj_nuts_kod VARCHAR(50),
    okres_lau_kod VARCHAR(50)
    );'''

    cursor.execute(sql)
    connection.commit()

    sql = '''CREATE TABLE suradnice(
    id SERIAL PRIMARY KEY,
    kraj VARCHAR(50),
    x FLOAT,
    y FLOAT,
    nazov VARCHAR(250)
    );'''

    cursor.execute(sql)
    connection.commit()

    sql = '''CREATE TABLE eu_countries(
    id SERIAL PRIMARY KEY,
    country VARCHAR(50),
    last_update VARCHAR(50),
    cases INTEGER,
    deaths INTEGER,
    recovered INTEGER
    );'''

    cursor.execute(sql)
    connection.commit()


def migrate_data(db):
    try:
        connection = psycopg2.connect(user="postgres",
                                      password="1234",
                                      host="127.0.0.1",
                                      port="5432",
                                      database="covid_cz")
        cursor = connection.cursor()

        drop1 = '''DROP TABLE IF EXISTS kraj_okres;'''
        drop2 = '''DROP TABLE IF EXISTS osoby;'''
        drop3 = '''DROP TABLE IF EXISTS umrti;'''
        drop4 = '''DROP TABLE IF EXISTS vyleceni;'''
        drop5 = '''DROP TABLE IF EXISTS suradnice;'''
        drop6 = '''DROP TABLE IF EXISTS eu_countries;'''

        cursor.execute(drop1)
        cursor.execute(drop2)
        cursor.execute(drop3)
        cursor.execute(drop4)
        cursor.execute(drop5)
        cursor.execute(drop6)

        connection.commit()

        create_tables(cursor, connection)

        sql = '''INSERT INTO kraj_okres(datum, kraj_nuts_kod, okres_lau_kod, kumulativni_pocet_nakazenych,
        kumulativni_pocet_vylecenych, kumulativni_pocet_umrti, x, y) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);'''
        tmp = db.kraj_okres.find()

        for i in tmp:
            cursor.execute(sql, (i['datum'], i['kraj_nuts_kod'], i['okres_lau_kod'], i['kumulativni_pocet_nakazenych'], i['kumulativni_pocet_vylecenych'], i['kumulativni_pocet_umrti'], i['x'], i['y']))
        connection.commit()

        print("kraj_okres")
        
        sql = '''INSERT INTO osoby(datum, vek, pohlavi, kraj_nuts_kod, okres_lau_kod) VALUES (%s, %s, %s, %s, %s);'''
        tmp = db.osoby.find()

        print("checkout")
        
        for i in tmp:
            cursor.execute(sql, (i['datum'], i['vek'], i['pohlavi'], i['kraj_nuts_kod'], i['okres_lau_kod']))
        connection.commit()

        print("osoby")

        sql = '''INSERT INTO umrti(datum, vek, pohlavi, kraj_nuts_kod, okres_lau_kod) VALUES (%s, %s, %s, %s, %s);'''
        tmp = db.umrti.find()

        for i in tmp:
            cursor.execute(sql, (i['datum'], i['vek'], i['pohlavi'], i['kraj_nuts_kod'], i['okres_lau_kod']))
        connection.commit()

        print("umrti")
        
        sql = '''INSERT INTO vyleceni(datum, vek, pohlavi, kraj_nuts_kod, okres_lau_kod) VALUES (%s, %s, %s, %s, %s);'''
        tmp = db.vyleceni.find()

        for i in tmp:
            cursor.execute(sql, (i['datum'], i['vek'], i['pohlavi'], i['kraj_nuts_kod'], i['okres_lau_kod']))
        connection.commit()

        print("vylieceni")
        sql = '''INSERT INTO suradnice(kraj, x, y, nazov) VALUES (%s, %s, %s, %s);'''
        tmp = db.suradnice.find()

        for i in tmp:
            cursor.execute(sql, (i['kraj'], i['x'], i['y'], i['nazov']))
        connection.commit()

        print("suradnice")
        
        sql = '''INSERT INTO eu_countries(country, last_update, cases, deaths, recovered) VALUES (%s, %s, %s, %s, %s);'''
        tmp = db.eu_countries.find()

        for i in tmp:
            cursor.execute(sql, (i['country'], i['last_update'], i['cases'], i['deaths'], i['recovered']))
        connection.commit()
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        # closing database connection.
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")


def normalize_data(col):
    data = db.osoby.find({})
    data_frame = pd.DataFrame(data)

    tmp = Quartile(data_frame, 'vek')
    tmp.fit()
    data_frame = tmp.transform(data_frame, 'vek')

    col.drop()
    col.insert_many(data_frame.to_dict('records'))


def request_function(db):
    URL = 'https://covid19-api.org/api/status'
    dates = db.kraj_okres.distinct('datum')
    eu_codes = ['AT', 'BE', 'BG', 'HR', 'CY', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IE', 'IT',
                'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE']

    db.eu_countries.drop()
    for code in eu_codes:
        for date in dates:
            r = requests.get(url=URL + '/' + code + '?date=' + date).json()
            if 'error' not in r:
                db.eu_countries.insert_one(r)


def add_coordinates(db):
    db.suradnice.drop()

    code_array = ['CZ010', 'CZ020', 'CZ031', 'CZ032', 'CZ041', 'CZ042', 'CZ051', 'CZ052', 'CZ053', 'CZ063',
                  'CZ064', 'CZ071', 'CZ072', 'CZ080']
    x_array = [50.0755, 49.8782, 48.9458, 49.4135, 50.1435, 50.6119, 50.6594, 50.3512, 49.9444, 49.4490,
               48.9545, 49.6587, 49.2162, 49.7305]
    y_array = [14.4378, 14.9363, 14.4416, 13.3157, 12.7502, 13.7870, 14.7632, 15.7976, 16.2857, 15.6406,
               16.7677, 17.0811, 17.7720, 18.2333]
    name_array = ['Hlavní město Praha', 'Středočeský kraj', 'Jihočeský kraj', 'Plzeňský kraj',
                  'Karlovarský kraj', 'Ústecký kraj', 'Liberecký kraj', 'Královéhradecký kraj',
                  'Pardubický kraj', 'Kraj Vysočina', 'Jihomoravský kraj', 'Olomoucký kraj',
                  'Zlínský kraj', 'Moravskoslezský kraj']

    for i in range(len(code_array)):
        db.suradnice.insert_one({'kraj': code_array[i], 'x': x_array[i], 'y': y_array[i], 'nazov': name_array[i]})

        db.kraj_okres.update_many({'kraj_nuts_kod': code_array[i]}, {'$set': {'x': x_array[i], 'y': y_array[i]}})


def create_collections(db):
    # ak existuju kolekcie tak ich zmazem
    db.kraj_okres.drop()
    db.osoby.drop()
    db.umrti.drop()
    db.vylieceni.drop()

    # vytvaram kolekcie
    kraj_okres_col = db.kraj_okres
    osoby_col = db.osoby
    umrti_col = db.umrti
    vyleceni_col = db.vyleceni

    return kraj_okres_col, osoby_col, umrti_col, vyleceni_col


def load_data():
    kraj_okres = pd.read_csv('data/kraj-okres-nakazeni-vyleceni-umrti.csv')
    osoby = pd.read_csv('data/osoby.csv')
    umrti = pd.read_csv('data/umrti.csv')
    vyleceni = pd.read_csv('data/vyleceni.csv')

    return kraj_okres, osoby, umrti, vyleceni


if __name__ == '__main__':
    print('Nacitavam data...')
    kraj_okres, osoby, umrti, vyleceni = load_data()

    print('Pripajam sa na server a vytvaram kolekcie...')
    client = MongoClient()
    db = client.covid_cz
    kraj_okres_col, osoby_col, umrti_col, vyleceni_col = create_collections(db)

    print('Konvertujem csv do jsonu...')
    kraj_okres = kraj_okres.to_dict('records')
    osoby = osoby.to_dict('records')
    umrti = umrti.to_dict('records')
    vyleceni = vyleceni.to_dict('records')

    print('Pridavam json do kolekcie...')
    kraj_okres_col.insert_many(kraj_okres)
    osoby_col.insert_many(osoby)
    umrti_col.insert_many(umrti)
    vyleceni_col.insert_many(vyleceni)

    print('Mazem atributy z kolekcie "osoby"...')
    osoby_col.update_many({}, {'$unset': {'nakaza_v_zahranici': 1, 'nakaza_zeme_csu_kod': 1}})

    print('Vytvaram kolekciu so suradnicami...')
    add_coordinates(db)

    print('Stahujem data z internetu...')
    request_function(db)

    print('Normalizujem data...')
    normalize_data(osoby_col)
    normalize_data(umrti_col)
    normalize_data(vyleceni_col)

    print('Migrujem data do relacnej databazy...')
    migrate_data(db)

    print('Hotovo')
