#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>
#include <nanobind/ndarray.h> // Include for nanobind::ndarray
#include <cstring> // For std::memcpy

namespace nb = nanobind;

namespace madEscape {

// This file creates the python bindings used by the learning code.
// Refer to the nanobind documentation for more details on these functions.
NB_MODULE(madrona_escape_room, m) {
    // Each simulator has a madrona submodule that includes base types
    // like madrona::py::Tensor and madrona::py::PyExecMode.
    madrona::py::setupMadronaSubmodule(m);

    // Expose Topo directly under the madrona_escape_room module
    nb::class_<Topo>(m, "Topo")
        .def(nb::init<>())
        .def_prop_rw("aj_link",
            [](Topo &self) {
                // Return a NumPy array referencing the C++ data
                return nb::ndarray<nb::numpy, uint32_t>(
                    self.aj_link, {100000, 5}, nb::handle());
            },
            [](Topo &self, nb::ndarray<> arr) {
                if (arr.ndim() != 2 || arr.shape(0) != 100000 || arr.shape(1) != 5)
                    throw std::invalid_argument("aj_link must have shape (100000, 5)");
                std::memcpy(self.aj_link, arr.data(), sizeof(uint32_t) * 100000 * 5);
            }
        )
        .def_rw("link_num", &Topo::link_num)
        .def_rw("net_npu_num", &Topo::net_npu_num);

    nb::class_<Manager> (m, "SimManager")
        .def("__init__", [](Manager *self,
                            madrona::py::PyExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            int64_t rand_seed,
                            bool auto_reset,
                            bool enable_batch_renderer,
                            uint32_t k_aray,
                            uint32_t cc_method,
                            nb::ndarray<> links,
                            Topo topo
                        ) 
        {
            if (links.ndim() != 2) {
                throw std::invalid_argument("Links must be a 2D array");
            }
            if (links.shape(0) != 2 || links.shape(1) != 100) {
                throw std::invalid_argument("Links must have shape (2, 100)");
            }

            // Convert the flat data pointer to a 2D array representation
            uint32_t **links_2d = new uint32_t*[links.shape(0)];
            for (size_t i = 0; i < links.shape(0); ++i) {
                links_2d[i] = static_cast<uint32_t *>(links.data()) + i * links.shape(1);
            }

            
            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .randSeed = (uint32_t)rand_seed,
                .autoReset = auto_reset,
                .enableBatchRenderer = enable_batch_renderer,
                .kAray = (uint32_t)k_aray, // fei add in 20241215
                .ccMethod = (uint32_t)cc_method,
                .topo = topo, // Reordered to match declaration order
                .Links = links_2d // Pass the 2D array pointer
            });
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("rand_seed"),
           nb::arg("auto_reset"),
           nb::arg("enable_batch_renderer") = false,
           nb::arg("k_aray") = 4, // fei add in 20241215
           nb::arg("cc_method") = 0,
           nb::arg("links"), // Update argument name
           nb::arg("topo")
        )   
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("self_observation_tensor", &Manager::selfObservationTensor)
        .def("partner_observations_tensor", &Manager::partnerObservationsTensor)
        .def("room_entity_observations_tensor",
             &Manager::roomEntityObservationsTensor)
        .def("door_observation_tensor",
             &Manager::doorObservationTensor)
        .def("lidar_tensor", &Manager::lidarTensor)
        .def("steps_remaining_tensor", &Manager::stepsRemainingTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
        .def("depth_tensor", &Manager::depthTensor)
        .def("results_tensor", &Manager::resultsTensor)  // fei add in 202412015
        .def("results2_tensor", &Manager::results2Tensor)
        .def("madronaEvents_tensor", &Manager::madronaEventsTensor)
        .def("madronaEventsResult_tensor", &Manager::madronaEventsResultTensor)
        .def("simulation_time_tensor", &Manager::simulationTimeTensor)
        .def("processParams_tensor",&Manager::processParamsTensor)
        .def("chakra_nodes_data_tensor", &Manager::chakraNodesDataTensor)
    ;
}

}
