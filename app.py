from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

# Helper functions to safely convert inputs
def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

@app.route("/assume", methods=["GET", "POST"])
def home():
    final_result = None

    if request.method == "POST":
        # Get form data safely
        data = CustomData(
            country_code=safe_int(request.form.get("country_code")),
            city=request.form.get("city", ""),
            locality='',
            cuisines=request.form.get("cuisines", ""),
            longitude=0.0,
            latitude=0.0,
            average_cost_for_two=safe_float(request.form.get("average_cost_for_two")),
            currency=request.form.get("currency", ""),
            has_table_booking=request.form.get("has_table_booking", "No"),
            has_online_delivery=request.form.get("has_online_delivery", "No"),
            is_delivering_now='',
            price_range=safe_int(request.form.get("price_range")),
            votes=safe_int(request.form.get("votes")),
            rating_color=''
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_frame()  # make sure this matches your class

        # Predict
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Round result
        final_result = round(float(results[0]), 2)

    return render_template("home.html", final_result=final_result)

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)





