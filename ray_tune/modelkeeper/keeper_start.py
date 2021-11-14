from config import modelkeeper_config
from matchingopt import ModelKeeper

keeper_service = ModelKeeper(modelkeeper_config)
keeper_service.start()

