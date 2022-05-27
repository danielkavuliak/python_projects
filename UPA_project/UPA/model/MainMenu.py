from flask import Flask,render_template,url_for,request,flash,redirect,jsonify, make_response
import plotly.express as px
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pandas as pd
import folium
import os
import pprint
import json
import matplotlib.pyplot as plt
import psycopg2

class Menu:
    def show_menu(self,db):
        try:
            connection = psycopg2.connect(user="postgres",
                                          password="1234",
                                          host="127.0.0.1",
                                          port="5432",
                                          database="covid_cz")

            cursor = connection.cursor()

            m = folium.Map(
                width=900, height=700,
                location=[49.8175, 15.4730],
                zoom_start=7,
                tiles='Stamen Terrain'
            )
            for kraj in db.suradnice_krajov.find():

                postgreSQL_select_Query = "select * from kraj_okres where kraj_okres.kraj_nuts_kod = kraj['kraj']"
                cursor.execute(postgreSQL_select_Query)
                query = pd.read_sql_query(postgreSQL_select_Query, connection)




                #query = db.kraj_okres.find( {"kraj_nuts_kod": kraj['kraj']})
                nak = []
                vyl = []
                umr = []
                num = 0
                for i,quer in query.iterrows():
                    nak.append(quer['kumulativni_pocet_nakazenych'])
                    vyl.append(quer['kumulativni_pocet_vylecenych'])
                    umr.append(quer['kumulativni_pocet_umrti'])
                    num+=1

                fig, ax = plt.subplots()
                plt.plot(nak)
                fig.savefig('templates/my_plot.png')
                html =f"""
                <h1>{kraj['nazov']} </h1><br>
                Priemer: {kraj['kraj']}
                <p>
                Priemerny pocet nakazenych: {sum(nak)/num}
                </p>
                <p>
                Priemerny pocet vyliecenych: {sum(vyl)/num}
                </p>
                </p>
                Priemerny pocet umrti:  {sum(umr)/num}
                </p>
                '<a href="http://127.0.0.1:5000/stats/{kraj['kraj']}"target="_blank"> Zobraz informacie o {kraj['nazov']}' </a>'
                """
                iframe = folium.IFrame(html=html, width=500, height=300)
                popup = folium.Popup(iframe, max_width=2650)

                folium.Marker(
                        location=[kraj['x'],kraj['y']],
                        popup= popup,
                        icon=folium.Icon(color='red', icon='info-sign')
                    ).add_to(m)

            outFileName = "templates/map.html"
            m.save(outfile=outFileName)
        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL", error)
        finally:
            # closing database connection.
            if (connection):
                cursor.close()
                connection.close()
                print("PostgreSQL connection is closed")

