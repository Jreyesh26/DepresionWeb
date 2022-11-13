import pandas as pd
import numpy as np
import sklearn
import joblib
from flask import Flask,render_template,request
app=Flask(__name__)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/result',methods=['GET','POST'])
def Enviar():
    if request.method == "POST":
        try:
            var_1 = float(request.form["P01"])
            var_2 = float(request.form["P02"])
            var_3 = float(request.form["P03"])
            var_4 = float(request.form["P04"])
            var_5 = float(request.form["P05"])
            var_6 = float(request.form["P06"])
            var_7 = float(request.form["P07"])
            var_8 = float(request.form["P08"])
            var_9 = float(request.form["P09"])
            var_10 = float(request.form["P10"])
            
            pred_args=[var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8,var_9,var_10]
            pred_arr=np.array(pred_args)
            preds=pred_arr.reshape(1, -1)
            modelo=open("./modeloNaiveBayes.pkl","rb")
            modelo_class=joblib.load(modelo)
            prediccion_modelo=modelo_class.predict(preds)
            prediccion_modelo=round(float(prediccion_modelo),2)
            if(prediccion_modelo == 1.0):
                prediccion_modelo="Usted posiblemente es alguien que tiene un cuadro de depresión"
                prediccion_m="Estos resultados no constituyen un diagnóstico. Puede hablar con un doctor o terapeuta para obtener un diagnóstico y/o acceder a terapia o medicamentos. Compartir estos resultados con alguien en quien confíe puede ser un excelente lugar para comenzar. Es recomendable acercarse a las oficinas de bienestar institucional de la Corporación Universitaria del Caribe, para darle seguimiento a su caso."
                estado="./static/img/triste.png"
            else:
                prediccion_modelo="Usted posiblemente es alguien que no tiene un cuadro de depresión"
                prediccion_m="Recuerde que estos resultados no son definitivos, si por alguna razón siente que necesita ayuda se le recomienda que se acerque a las oficinas de bienestar institucional  de la Corporación Universitaria del Caribe, para hacerle seguimiento a su caso. De lo contrario continue normalmente con su vida"
                estado="./static/img/carafeliz.png"
        except ValueError:
                return "Por Favor Ingrese Datos Validos"
        return render_template("result.html", prediccion = prediccion_modelo, prediccion2=prediccion_m, prediccion3=estado)


if __name__ == "__main__":
    app.run(debug=True)