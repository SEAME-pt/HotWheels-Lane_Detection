#include "../../includes/objectDetection/LabelManager.hpp"

/**
 * @brief Construtor que carrega as labels de um arquivo.
 * @param labelPath Caminho para o arquivo de labels.
 */
LabelManager::LabelManager(const std::string& labelPath) {
	loadLabels(labelPath);
}

/**
 * @brief Carrega as labels do arquivo especificado.
 * @param labelPath Caminho para o arquivo de labels.
 */
void LabelManager::loadLabels(const std::string& labelPath) {
	std::ifstream file(labelPath);
	if (!file.is_open()) {
		std::cerr << "[ERRO] Não foi possível abrir o arquivo de labels: " << labelPath << std::endl;
		return;
	}

	std::string line;
	while (std::getline(file, line)) {
		line.erase(0, line.find_first_not_of(" \t\r\n"));
		line.erase(line.find_last_not_of(" \t\r\n") + 1);
		labels.push_back(line);
	}
	file.close();

	std::cout << "[INFO] Carregadas " << labels.size() << " labels." << std::endl;
}

/**
 * @brief Retorna o nome da label para um dado classId.
 * @param classId Índice da classe.
 * @return Nome da classe ou "Unknown".
 */
std::string LabelManager::getLabel(int classId) const {
	// Correção: cast para evitar warning signed/unsigned
	if (classId >= 0 && static_cast<size_t>(classId) < labels.size()) {
		return labels[classId];
	}
	return "Unknown";
}

/**
 * @brief Retorna o número de classes carregadas.
 * @return Número de classes.
 */
size_t LabelManager::getNumClasses() const {
	return labels.size();
}
