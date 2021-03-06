add_requires("kompute", "opencv", "nlohmann_json")

includes("test")

target("lgmd")
	set_kind("binary")
	add_languages("cxx17")
	add_rules("utils.glsl2spv", {bin2c = true})
	add_rules("utils.bin2c", {extensions = {".spv"}})
	add_includedirs("./src")
	add_files("./src/*.cpp", "./rsc/*.comp")
	add_packages("kompute", "opencv", "nlohmann_json")