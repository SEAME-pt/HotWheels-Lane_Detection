/*!
 * @file enums.hpp
 * @brief Definition of the enums used in the application.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the definition of the enums used in the
 * application.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef ENUMS_HPP
#define ENUMS_HPP

#include <QtCore/qmetatype.h>

/*! @brief Enum class for the component status. */
enum class ComponentStatus { Idle, Starting, Operational, Down };
/*! @brief Enum class for the driving mode. */
enum class DrivingMode { Manual, Automatic };
/*! @brief Enum class for the cluster theme. */
enum class ClusterTheme { Dark, Light };
/*! @brief Enum class for the cluster metrics. */
enum class ClusterMetrics { Miles, Kilometers };
/*! @brief Enum class for the car direction. */
enum class CarDirection { Drive, Reverse, Stop };

Q_DECLARE_METATYPE(ComponentStatus)
Q_DECLARE_METATYPE(DrivingMode)
Q_DECLARE_METATYPE(ClusterTheme)
Q_DECLARE_METATYPE(ClusterMetrics)
Q_DECLARE_METATYPE(CarDirection)

#endif // ENUMS_HPP
