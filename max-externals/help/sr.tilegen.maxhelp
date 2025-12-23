{
	"patcher": {
		"fileversion": 1,
		"appversion": {
			"major": 8,
			"minor": 6,
			"revision": 0,
			"architecture": "x64",
			"modernui": 1
		},
		"classnamespace": "box",
		"rect": [
			100.0,
			100.0,
			800.0,
			600.0
		],
		"description": "Random tile generator for bandswap and corruption effects",
		"digest": "Generates random tile bounds with channel permutations",
		"tags": "jitter, tiles, random, bandswap, corruption",
		"boxes": [
			{
				"box": {
					"id": "obj-1",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						30.0,
						20.0,
						400.0,
						20.0
					],
					"text": "sr.tilegen - Random Tile Generator (CPU External)",
					"fontsize": 14.0,
					"fontface": 1
				}
			},
			{
				"box": {
					"id": "obj-2",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						30.0,
						45.0,
						600.0,
						60.0
					],
					"text": "Generates random tile bounds for sr.bandswap and sr.corruption effects.\nOutput: list of (x y w h perm) tuples for each tile.\nUse with jit.fill to create mask texture for GPU shaders."
				}
			},
			{
				"box": {
					"id": "obj-world",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						550.0,
						130.0,
						150.0,
						22.0
					],
					"text": "jit.world sr_tilegen_ctx @visible 0"
				}
			},
			{
				"box": {
					"id": "obj-3",
					"maxclass": "button",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"bang"
					],
					"patching_rect": [
						30.0,
						120.0,
						24.0,
						24.0
					]
				}
			},
			{
				"box": {
					"id": "obj-4",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						400.0,
						120.0,
						120.0,
						20.0
					],
					"text": "tile_count: 1-1000"
				}
			},
			{
				"box": {
					"id": "obj-dial-count",
					"maxclass": "dial",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"float"
					],
					"patching_rect": [
						400.0,
						145.0,
						40.0,
						40.0
					],
					"size": 1000.0,
					"min": 0.0,
					"mult": 1.0
				}
			},
			{
				"box": {
					"id": "obj-expr-count",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						450.0,
						155.0,
						55.0,
						22.0
					],
					"text": "expr $f1+1"
				}
			},
			{
				"box": {
					"id": "obj-5",
					"maxclass": "number",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						510.0,
						155.0,
						50.0,
						22.0
					],
					"minimum": 1,
					"maximum": 1000
				}
			},
			{
				"box": {
					"id": "obj-6",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						510.0,
						185.0,
						100.0,
						22.0
					],
					"text": "tile_count $1"
				}
			},
			{
				"box": {
					"id": "obj-7",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						400.0,
						210.0,
						120.0,
						20.0
					],
					"text": "tile_size_min: 0.01-1.0"
				}
			},
			{
				"box": {
					"id": "obj-dial-min",
					"maxclass": "dial",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"float"
					],
					"patching_rect": [
						400.0,
						235.0,
						40.0,
						40.0
					],
					"size": 100.0,
					"min": 0.0,
					"mult": 0.01
				}
			},
			{
				"box": {
					"id": "obj-8",
					"maxclass": "flonum",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						450.0,
						250.0,
						60.0,
						22.0
					],
					"minimum": 0.01,
					"maximum": 1.0
				}
			},
			{
				"box": {
					"id": "obj-9",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						450.0,
						280.0,
						100.0,
						22.0
					],
					"text": "tile_size_min $1"
				}
			},
			{
				"box": {
					"id": "obj-10",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						560.0,
						210.0,
						130.0,
						20.0
					],
					"text": "tile_size_max: 0.01-1.0"
				}
			},
			{
				"box": {
					"id": "obj-dial-max",
					"maxclass": "dial",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"float"
					],
					"patching_rect": [
						560.0,
						235.0,
						40.0,
						40.0
					],
					"size": 100.0,
					"min": 0.0,
					"mult": 0.01
				}
			},
			{
				"box": {
					"id": "obj-11",
					"maxclass": "flonum",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						610.0,
						250.0,
						60.0,
						22.0
					],
					"minimum": 0.01,
					"maximum": 1.0
				}
			},
			{
				"box": {
					"id": "obj-12",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						610.0,
						280.0,
						100.0,
						22.0
					],
					"text": "tile_size_max $1"
				}
			},
			{
				"box": {
					"id": "obj-13",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						30.0,
						220.0,
						350.0,
						22.0
					],
					"text": "sr.tilegen @tile_count 10 @tile_size_min 0.05 @tile_size_max 0.2"
				}
			},
			{
				"box": {
					"id": "obj-14",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						30.0,
						280.0,
						700.0,
						22.0
					],
					"text": ""
				}
			},
			{
				"box": {
					"id": "obj-15",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						30.0,
						330.0,
						700.0,
						80.0
					],
					"text": "Output format: x y w h perm x y w h perm ...\n- x, y: tile position in pixels\n- w, h: tile dimensions in pixels\n- perm: channel permutation index (0-5)\n  0=RGB, 1=RBG, 2=GRB, 3=GBR, 4=BRG, 5=BGR"
				}
			},
			{
				"box": {
					"id": "obj-16",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						30.0,
						420.0,
						500.0,
						40.0
					],
					"text": "Uses PCG-based RNG for deterministic, reproducible tile generation.\nSame seed produces identical tile layout across sessions."
				}
			}
		],
		"lines": [
			{
				"patchline": {
					"source": [
						"obj-3",
						0
					],
					"destination": [
						"obj-13",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-dial-count",
						0
					],
					"destination": [
						"obj-expr-count",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-expr-count",
						0
					],
					"destination": [
						"obj-5",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-5",
						0
					],
					"destination": [
						"obj-6",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-6",
						0
					],
					"destination": [
						"obj-13",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-dial-min",
						0
					],
					"destination": [
						"obj-8",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-8",
						0
					],
					"destination": [
						"obj-9",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-9",
						0
					],
					"destination": [
						"obj-13",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-dial-max",
						0
					],
					"destination": [
						"obj-11",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-11",
						0
					],
					"destination": [
						"obj-12",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-12",
						0
					],
					"destination": [
						"obj-13",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-13",
						0
					],
					"destination": [
						"obj-14",
						0
					]
				}
			}
		]
	}
}