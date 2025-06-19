#include "../../includes/inference/InferenceManager.hpp"
#include <algorithm>
#include <experimental/filesystem>
#include <iostream>

InferenceManager::InferenceManager()
	: current_model_type_(ModelType::AUTO_DETECT) {}

InferenceManager::~InferenceManager() { clear(); }

ModelType InferenceManager::detectModelType(const std::string &model_path)
{
	// Obter extensão do arquivo
	std::experimental::filesystem::path path(model_path);
	std::string extension = path.extension().string();

	// Converter para lowercase
	std::transform(extension.begin(), extension.end(), extension.begin(),
				   ::tolower);

	if (extension == ".onnx" || extension == ".pt")
	{
		return ModelType::PYTORCH_ONNX;
	}
	else if (extension == ".h5" || path.filename() == "saved_model.pb")
	{
		return ModelType::KERAS;
	}
	else
	{
		std::cerr << "[InferenceManager] Aviso: Extensão desconhecida '"
				  << extension << "'. Tentando ONNX por padrão." << std::endl;
		return ModelType::PYTORCH_ONNX;
	}
}

bool InferenceManager::loadModel(const std::string &model_name,
								 const std::string &model_path,
								 ModelType model_type)
{

	// Verificar se arquivo existe
	if (!std::experimental::filesystem::exists(model_path))
	{
		std::cerr << "[InferenceManager] Erro: Arquivo do modelo não encontrado: "
				  << model_path << std::endl;
		return false;
	}

	// Detectar tipo automaticamente se necessário
	if (model_type == ModelType::AUTO_DETECT)
	{
		model_type = detectModelType(model_path);
	}

	// Criar inferenciador apropriado
	std::unique_ptr<IInferencer> inferencer;

	try
	{
		switch (model_type)
		{
		case ModelType::PYTORCH_ONNX:
			std::cout << "[InferenceManager] Carregando modelo PyTorch/ONNX: "
					  << model_name << std::endl;
			// inferencer = std::make_unique<ONNXInferencer>(model_path);
			break;

		case ModelType::KERAS:
			std::cout << "[InferenceManager] Carregando modelo Keras: " << model_name
					  << std::endl;
			inferencer = std::make_unique<KerasInferencer>(model_path);
			break;

		default:
			std::cerr << "[InferenceManager] Erro: Tipo de modelo não suportado"
					  << std::endl;
			return false;
		}

		if (!inferencer)
		{
			std::cerr << "[InferenceManager] Erro: Falha ao criar inferenciador"
					  << std::endl;
			return false;
		}

		// Armazenar inferenciador
		inferencers_[model_name] = std::move(inferencer);

		// Se é o primeiro modelo, torná-lo ativo
		if (current_model_name_.empty())
		{
			current_model_name_ = model_name;
			current_model_type_ = model_type;
		}

		std::cout << "[InferenceManager] Modelo '" << model_name
				  << "' carregado com sucesso!" << std::endl;
		return true;
	}
	catch (const std::exception &e)
	{
		std::cerr << "[InferenceManager] Erro ao carregar modelo '" << model_name
				  << "': " << e.what() << std::endl;
		return false;
	}
}

bool InferenceManager::selectModel(const std::string &model_name)
{
	auto it = inferencers_.find(model_name);
	if (it == inferencers_.end())
	{
		std::cerr << "[InferenceManager] Erro: Modelo '" << model_name
				  << "' não está carregado" << std::endl;
		return false;
	}

	current_model_name_ = model_name;
	std::cout << "[InferenceManager] Modelo ativo alterado para: " << model_name
			  << std::endl;
	return true;
}

cv::Mat InferenceManager::predict(const cv::Mat &image)
{
	if (current_model_name_.empty())
	{
		std::cerr << "[InferenceManager] Erro: Nenhum modelo ativo" << std::endl;
		return cv::Mat();
	}

	return predict(current_model_name_, image);
}

cv::Mat InferenceManager::predict(const std::string &model_name,
								  const cv::Mat &image)
{
	auto it = inferencers_.find(model_name);
	if (it == inferencers_.end())
	{
		std::cerr << "[InferenceManager] Erro: Modelo '" << model_name
				  << "' não está carregado" << std::endl;
		return cv::Mat();
	}

	try
	{
		// Para compatibilidade com a interface atual, vamos usar doInference
		// Em uma implementação real, você modificaria IInferencer para ter um
		// método predict
		it->second->doInference(image);

		// Por enquanto, retornar a imagem original como placeholder
		// Você deve modificar IInferencer para retornar cv::Mat do predict
		return image.clone();
	}
	catch (const std::exception &e)
	{
		std::cerr << "[InferenceManager] Erro durante predição com modelo '"
				  << model_name << "': " << e.what() << std::endl;
		return cv::Mat();
	}
}

bool InferenceManager::isModelLoaded(const std::string &model_name) const
{
	return inferencers_.find(model_name) != inferencers_.end();
}

std::vector<std::string> InferenceManager::getLoadedModels() const
{
	std::vector<std::string> models;
	for (const auto &pair : inferencers_)
	{
		models.push_back(pair.first);
	}
	return models;
}

void InferenceManager::unloadModel(const std::string &model_name)
{
	auto it = inferencers_.find(model_name);
	if (it != inferencers_.end())
	{
		inferencers_.erase(it);
		std::cout << "[InferenceManager] Modelo '" << model_name
				  << "' removido da memória" << std::endl;

		// Se era o modelo ativo, limpar
		if (current_model_name_ == model_name)
		{
			current_model_name_.clear();
			if (!inferencers_.empty())
			{
				// Selecionar o primeiro modelo disponível
				current_model_name_ = inferencers_.begin()->first;
				std::cout << "[InferenceManager] Modelo ativo alterado para: "
						  << current_model_name_ << std::endl;
			}
		}
	}
}

void InferenceManager::clear()
{
	inferencers_.clear();
	current_model_name_.clear();
	std::cout << "[InferenceManager] Todos os modelos removidos da memória"
			  << std::endl;
}
