#include <gtest/gtest.h>
#include "../includes/objectDetection/LabelManager.hpp"

// Helper to create a temporary label file for testing
std::string createTempLabelFile(const std::vector<std::string>& labels) {
    std::string filePath = "test_labels.txt";
    std::ofstream file(filePath);
    for (const auto& label : labels) {
        file << label << "\n";
    }
    file.close();
    return filePath;
}

TEST(LabelManagerTest, LoadLabelsCorrectly) {
    std::vector<std::string> testLabels = {"car", "person", "bicycle"};
    std::string filePath = createTempLabelFile(testLabels);

    LabelManager manager(filePath);

    ASSERT_EQ(manager.getNumClasses(), testLabels.size());
    for (size_t i = 0; i < testLabels.size(); ++i) {
        EXPECT_EQ(manager.getLabel(i), testLabels[i]);
    }

    // Clean up
    std::remove(filePath.c_str());
}

TEST(LabelManagerTest, InvalidClassIdReturnsUnknown) {
    std::vector<std::string> testLabels = {"cat", "dog"};
    std::string filePath = createTempLabelFile(testLabels);

    LabelManager manager(filePath);

    EXPECT_EQ(manager.getLabel(-1), "Unknown");
    EXPECT_EQ(manager.getLabel(100), "Unknown");

    std::remove(filePath.c_str());
}

TEST(LabelManagerTest, HandlesNonExistentFile) {
    LabelManager manager("non_existent_file.txt");
    EXPECT_EQ(manager.getNumClasses(), 0);
    EXPECT_EQ(manager.getLabel(0), "Unknown");
}

TEST(LabelManagerTest, HandlesEmptyFile) {
	std::string filePath = "empty_labels.txt";
	std::ofstream file(filePath);
	file.close();

	LabelManager manager(filePath);

	EXPECT_EQ(manager.getNumClasses(), 0);
	EXPECT_EQ(manager.getLabel(0), "Unknown");

	std::remove(filePath.c_str());
}

