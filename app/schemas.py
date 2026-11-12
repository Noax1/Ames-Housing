from pydantic import BaseModel, ConfigDict, Field


class HouseFeaturesRequest(BaseModel):
    # Разрешаем передавать остальные 71 признак из датасета (включая null/None)
    model_config = ConfigDict(extra="allow")

    OverallQual: int = Field(..., ge=1, le=10, description="Общая оценка качества материалов и отделки (1-10)")
    GrLivArea: float = Field(..., gt=0, description="Жилая площадь над уровнем земли (кв. футы)")
    TotalBsmtSF: float = Field(..., ge=0, description="Общая площадь подвала (кв. футы)")
    GarageCars: int = Field(..., ge=0, le=10, description="Вместимость гаража (кол-во машин)")
    YearBuilt: int = Field(..., ge=1800, le=2026, description="Год постройки дома")
    Neighborhood: str = Field(..., min_length=1, description="Район города Эймс (например, CollgCr)")
    LotArea: float = Field(..., gt=0, description="Площадь участка (кв. футы)")
    YrSold: int = Field(..., ge=2006, le=2026, description="Год продажи")


class PredictionResponse(BaseModel):
    predicted_price_usd: float
    price_segment: str
    model_version: str