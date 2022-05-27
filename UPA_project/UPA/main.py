from flask import Flask,render_template,url_for,request,flash,redirect,jsonify, make_response, send_from_directory
from model import *
from model import MainMenu
from pymongo import MongoClient
import pymongo
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pandas as pd
import matplotlib as plt
import os
import json
import seaborn as sns
import psycopg2



client = MongoClient()
db = client['mongodb_connection']

app = Flask(__name__,static_url_path='/static',static_folder= "templates/static")
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route('/<path:filename>')
def send_file(filename):
    return send_from_directory(app.static_folder, filename)


@app.route('/stats/<kraj_nuts_kod>',methods = ['POST','GET'])
def show_more(kraj_nuts_kod):
    try:
        connection = psycopg2.connect(user="postgres",
                                      password="1234",
                                      host="127.0.0.1",
                                      port="5432",
                                      database="covid_cz")

        cursor = connection.cursor()

        print(kraj_nuts_kod)
        postgreSQL_select_Query = """select * from kraj_okres where kraj_okres.kraj_nuts_kod = '%s'"""
        cursor.execute("""select * from kraj_okres where kraj_okres.kraj_nuts_kod = '%s'""" % (kraj_nuts_kod,))
        data_frame_okres = pd.read_sql_query(postgreSQL_select_Query % (kraj_nuts_kod,), connection)
        print("**********************")
        print(data_frame_okres)

        postgreSQL_select_Query = """select * from osoby where osoby.kraj_nuts_kod  = '%s'"""
        cursor.execute("""select * from osoby where osoby.kraj_nuts_kod = '%s'""" % (kraj_nuts_kod,))
        data_frame_osoby = pd.read_sql_query(postgreSQL_select_Query % (kraj_nuts_kod,), connection)


        postgreSQL_select_Query = """select * from umrti where umrti.kraj_nuts_kod  = '%s'"""
        cursor.execute("""select * from umrti where umrti.kraj_nuts_kod = '%s'""" % (kraj_nuts_kod,))
        data_frame_zomreli = pd.read_sql_query(postgreSQL_select_Query % (kraj_nuts_kod,), connection)

        postgreSQL_select_Query = """select * from vyleceni where vyleceni.kraj_nuts_kod  = '%s'"""
        cursor.execute("""select * from vyleceni where vyleceni.kraj_nuts_kod = '%s'""" % (kraj_nuts_kod,))
        data_frame_vyleceni = pd.read_sql_query(postgreSQL_select_Query % (kraj_nuts_kod,), connection)


        fig, ax = plt.pyplot.subplots(figsize=(15,15))
        data_frame_osoby['vek'].value_counts().plot.bar()
        ax.title.set_text('Histogram zobrazujuci vek ludi s COVID')
        kod = str(kraj_nuts_kod.lower())
        fig.savefig('templates/static/images/my_plot_osoby_'+str(kraj_nuts_kod.lower())+'.png')

        fig2, ax2 = plt.pyplot.subplots(figsize=(15,15))
        sns.boxplot(x=data_frame_osoby['pohlavi'], y=data_frame_osoby['vek'])
        ax2.title.set_text('Pohlavie ludi s COVID vykreslene pomocou boxplotou')
        fig2.savefig('templates/static/images/my_plot_osoby_pohl_'+str(kraj_nuts_kod.lower())+'.png')


        fig3, ax3 = plt.pyplot.subplots(figsize=(15,15))
        sns.boxplot(x=data_frame_vyleceni['pohlavi'], y=data_frame_vyleceni['vek'])
        ax3.title.set_text('Pohlavie a vek vyliecenych ludi s COVID vykreslene pomocou boxplotou')
        fig3.savefig('templates/static/images/my_plot_vylieceni_'+str(kraj_nuts_kod.lower())+'.png')

        fig4, ax4 = plt.pyplot.subplots(figsize=(15, 15))
        sns.boxplot(x=data_frame_zomreli['pohlavi'], y=data_frame_zomreli['vek'])
        ax4.title.set_text('Pohlavie a vek zomrelych ludi s COVID vykreslene pomocou boxplotou')
        fig4.savefig('templates/static/images/my_plot_zomreli_' + str(kraj_nuts_kod.lower()) + '.png')

        nakazeny_dokopy = data_frame_okres.groupby('datum')['kumulativni_pocet_nakazenych'].sum()
        vylieceny_dokopy = data_frame_okres.groupby('datum')['kumulativni_pocet_vylecenych'].sum()
        umrti_dokopy = data_frame_okres.groupby('datum')['kumulativni_pocet_umrti'].sum()

        fig5, ax5 = plt.pyplot.subplots(figsize=(50, 40))
        lineplot = sns.lineplot(data=[nakazeny_dokopy, vylieceny_dokopy, umrti_dokopy])
        for item in lineplot.get_xticklabels():
            item.set_rotation(90)
        ax5.title.set_text('Vyvoj COVID  v case')
        fig5.savefig('templates/static/images/my_plot_vyvoj_' + str(kraj_nuts_kod.lower()) + '.png')

        descriptic = data_frame_osoby.describe(include=['object'], exclude=['int64', 'float64'])
        #print(descriptic)

        query = db.suradnice_krajov.find({"kraj": kraj_nuts_kod})
        for element in query:
            data = element
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        # closing database connection.
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
    return render_template('kraj.html', kraj_nazov=data['nazov'],kod=kod,tables=[descriptic.to_html(classes='data')], titles=descriptic.columns.values)



@app.route('/',methods = ['POST','GET'])
def show_menu():
    if request.method == 'POST':
        if request.form['button'] == 'Zobraiť mapu covidu':
            MainMenu.Menu().show_menu(db)
            return render_template('map.html')
        elif request.form['button'] == 'Zobraziť štatistiky':
            return redirect('/global_stats')
    return render_template('home.html')

@app.route('/global_stats',methods = ['POST','GET'])
def show_stats():

    try:
        connection = psycopg2.connect(user="postgres",
                                      password="1234",
                                      host="127.0.0.1",
                                      port="5432",
                                      database="covid_cz")

        cursor = connection.cursor()

        postgreSQL_select_Query = "select * from osoby"
        cursor.execute(postgreSQL_select_Query)
        data_frame_osoby = pd.read_sql_query(postgreSQL_select_Query, connection)

        postgreSQL_select_Query = "select * from kraj_okres"
        cursor.execute(postgreSQL_select_Query)
        data_frame_okres = pd.read_sql_query(postgreSQL_select_Query, connection)

        postgreSQL_select_Query = "select * from umrti"
        cursor.execute(postgreSQL_select_Query)
        data_frame_zomreli = pd.read_sql_query(postgreSQL_select_Query, connection)

        postgreSQL_select_Query = "select * from vyleceni"
        cursor.execute(postgreSQL_select_Query)
        data_frame_vyleceni = pd.read_sql_query(postgreSQL_select_Query, connection)

        postgreSQL_select_Query = "select * from suradnice"
        cursor.execute(postgreSQL_select_Query)
        data_frame_suradnice = pd.read_sql_query(postgreSQL_select_Query, connection)


        print(data_frame_osoby)

        descriptic = data_frame_osoby.describe(include=['object'], exclude=['int64', 'float64'])

        descriptic2 = data_frame_okres.describe()

        descriptic3 = data_frame_zomreli.describe()

        descriptic4 = data_frame_vyleceni.describe()


        nakazeny_dokopy = data_frame_okres.groupby('datum')['kumulativni_pocet_nakazenych'].sum()
        vylieceny_dokopy = data_frame_okres.groupby('datum')['kumulativni_pocet_vylecenych'].sum()
        umrti_dokopy = data_frame_okres.groupby('datum')['kumulativni_pocet_umrti'].sum()

        fig, ax = plt.pyplot.subplots(figsize=(50, 40))
        lineplot = sns.lineplot(data=[nakazeny_dokopy, vylieceny_dokopy, umrti_dokopy])
        for item in lineplot.get_xticklabels():
            item.set_rotation(90)
        ax.title.set_text('Vyvoj COVID  v case')
        fig.savefig('templates/static/images/my_plot_vyvoj_celkovy.png')


        fig2, ax2 = plt.pyplot.subplots(figsize=(15,15))
        sns.boxplot(x=data_frame_vyleceni['pohlavi'], y=data_frame_vyleceni['vek'])
        ax2.title.set_text('Pohlavie a vek vyliecenych ludi s COVID vykreslene pomocou boxplotou')
        fig2.savefig('templates/static/images/my_plot_vylieceni_celkovy.png')

        fig3, ax3 = plt.pyplot.subplots(figsize=(15,15))
        data_frame_okres['kraj_nuts_kod'].value_counts().plot.bar()
        ax3.title.set_text('Pocet nakazenych v jednotlivych krajoch')
        fig3.savefig('templates/static/images/my_plot_zastupenie_celkovy.png')

        kod_nazov = data_frame_suradnice[["kraj", "nazov"]]
        print(descriptic)
        print([descriptic.to_html(classes='data')])
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        # closing database connection.
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
    return render_template('global_stats.html',tables=[descriptic.to_html(classes='data')], titles=descriptic.columns.values,tables2 = [descriptic2.to_html(classes='data')], titles2=descriptic2.columns.values,tables3 = [descriptic3.to_html(classes='data')], titles3=descriptic3.columns.values,tables4 = [descriptic4.to_html(classes='data')], titles4=descriptic4.columns.values,tables5 = [kod_nazov.to_html(classes='data')], titles5=kod_nazov.columns.values)

if __name__ == "__main__":
    app.run(debug=True)