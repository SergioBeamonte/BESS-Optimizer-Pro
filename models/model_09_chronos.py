import torch

def forecast_chronos(pipeline, series, steps_ahead):
    context = torch.tensor(series.values, dtype=torch.float32)
    forecast = pipeline.predict(context.unsqueeze(0), prediction_length=steps_ahead, num_samples=20)
    return forecast[0].median(dim=0)[0].numpy()
