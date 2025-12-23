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
			850.0,
			650.0
		],
		"description": "Buffer corruption effects simulating cosmic ray memory upsets",
		"digest": "XOR, invert, and shuffle corruption in tile regions",
		"tags": "jitter, GPU, corruption, glitch, satellite, effect",
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
						500.0,
						20.0
					],
					"text": "sr.corruption - Buffer Corruption Effects",
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
						700.0,
						40.0
					],
					"text": "Simulates cosmic ray hits causing single-event upsets in satellite memory.\nWorks with tile mask from sr.tilegen. Three corruption modes available."
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
					"text": "jit.world sr_corruption_ctx @visible 0"
				}
			},
			{
				"box": {
					"id": "obj-3",
					"maxclass": "toggle",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"int"
					],
					"patching_rect": [
						30.0,
						100.0,
						24.0,
						24.0
					]
				}
			},
			{
				"box": {
					"id": "obj-4",
					"maxclass": "newobj",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						"bang"
					],
					"patching_rect": [
						30.0,
						130.0,
						65.0,
						22.0
					],
					"text": "qmetro 30"
				}
			},
			{
				"box": {
					"id": "obj-5",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"jit_gl_texture",
						""
					],
					"patching_rect": [
						30.0,
						160.0,
						380.0,
						22.0
					],
					"text": "jit.movie @autostart 1 @loop 1 @output_texture 1 @drawto sr_corruption_ctx"
				}
			},
			{
				"box": {
					"id": "obj-20",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"bang"
					],
					"patching_rect": [
						400.0,
						100.0,
						58.0,
						22.0
					],
					"text": "loadbang"
				}
			},
			{
				"box": {
					"id": "obj-delay",
					"maxclass": "newobj",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						"bang"
					],
					"patching_rect": [
						250.0,
						130.0,
						63.0,
						22.0
					],
					"text": "delay 100"
				}
			},
			{
				"box": {
					"id": "obj-21",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						250.0,
						160.0,
						120.0,
						22.0
					],
					"text": "read chickens.mp4"
				}
			},
			{
				"box": {
					"id": "obj-noise",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"jit_matrix",
						""
					],
					"patching_rect": [
						200.0,
						200.0,
						120.0,
						22.0
					],
					"text": "jit.noise 4 char 16 16"
				}
			},
			{
				"box": {
					"id": "obj-op",
					"maxclass": "newobj",
					"numinlets": 2,
					"numoutlets": 2,
					"outlettype": [
						"jit_matrix",
						""
					],
					"patching_rect": [
						200.0,
						230.0,
						120.0,
						22.0
					],
					"text": "jit.op @op > @val 128"
				}
			},
			{
				"box": {
					"id": "obj-tex",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"jit_gl_texture",
						""
					],
					"patching_rect": [
						200.0,
						260.0,
						250.0,
						22.0
					],
					"text": "jit.gl.texture sr_corruption_ctx @name mask"
				}
			},
			{
				"box": {
					"id": "obj-6",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						450.0,
						200.0,
						80.0,
						20.0
					],
					"text": "mode (click)"
				}
			},
			{
				"box": {
					"id": "obj-mode-btn",
					"maxclass": "button",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"bang"
					],
					"patching_rect": [
						450.0,
						225.0,
						24.0,
						24.0
					]
				}
			},
			{
				"box": {
					"id": "obj-mode-counter",
					"maxclass": "newobj",
					"numinlets": 5,
					"numoutlets": 4,
					"outlettype": [
						"int",
						"",
						"",
						"int"
					],
					"patching_rect": [
						450.0,
						255.0,
						80.0,
						22.0
					],
					"text": "counter 0 2"
				}
			},
			{
				"box": {
					"id": "obj-mode-select",
					"maxclass": "newobj",
					"numinlets": 4,
					"numoutlets": 4,
					"outlettype": [
						"bang",
						"bang",
						"bang",
						""
					],
					"patching_rect": [
						450.0,
						285.0,
						80.0,
						22.0
					],
					"text": "select 0 1 2"
				}
			},
			{
				"box": {
					"id": "obj-m0",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						450.0,
						315.0,
						55.0,
						22.0
					],
					"text": "mode 0"
				}
			},
			{
				"box": {
					"id": "obj-m1",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						450.0,
						340.0,
						55.0,
						22.0
					],
					"text": "mode 1"
				}
			},
			{
				"box": {
					"id": "obj-m2",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						450.0,
						365.0,
						55.0,
						22.0
					],
					"text": "mode 2"
				}
			},
			{
				"box": {
					"id": "obj-n0",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						510.0,
						315.0,
						50.0,
						22.0
					],
					"text": "set XOR"
				}
			},
			{
				"box": {
					"id": "obj-n1",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						510.0,
						340.0,
						50.0,
						22.0
					],
					"text": "set Inv"
				}
			},
			{
				"box": {
					"id": "obj-n2",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						510.0,
						365.0,
						50.0,
						22.0
					],
					"text": "set Shuf"
				}
			},
			{
				"box": {
					"id": "obj-mode-name",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						480.0,
						227.0,
						50.0,
						20.0
					],
					"text": "XOR"
				}
			},
			{
				"box": {
					"id": "obj-7",
					"maxclass": "number",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						450.0,
						395.0,
						50.0,
						22.0
					],
					"minimum": 0,
					"maximum": 2
				}
			},
			{
				"box": {
					"id": "obj-8",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						450.0,
						425.0,
						80.0,
						22.0
					],
					"text": "mode $1"
				}
			},
			{
				"box": {
					"id": "obj-9",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						570.0,
						200.0,
						100.0,
						20.0
					],
					"text": "intensity: 0.0-1.0"
				}
			},
			{
				"box": {
					"id": "obj-dial-intensity",
					"maxclass": "dial",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"float"
					],
					"patching_rect": [
						570.0,
						225.0,
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
					"id": "obj-10",
					"maxclass": "flonum",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						620.0,
						240.0,
						60.0,
						22.0
					],
					"minimum": 0.0,
					"maximum": 1.0
				}
			},
			{
				"box": {
					"id": "obj-11",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						620.0,
						270.0,
						80.0,
						22.0
					],
					"text": "intensity $1"
				}
			},
			{
				"box": {
					"id": "obj-12",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						700.0,
						200.0,
						80.0,
						20.0
					],
					"text": "seed: 0-1000"
				}
			},
			{
				"box": {
					"id": "obj-dial-seed",
					"maxclass": "dial",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"float"
					],
					"patching_rect": [
						700.0,
						225.0,
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
					"id": "obj-13",
					"maxclass": "number",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						750.0,
						240.0,
						50.0,
						22.0
					],
					"minimum": 0
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
						750.0,
						270.0,
						60.0,
						22.0
					],
					"text": "seed $1"
				}
			},
			{
				"box": {
					"id": "obj-15",
					"maxclass": "newobj",
					"numinlets": 2,
					"numoutlets": 2,
					"outlettype": [
						"jit_gl_texture",
						""
					],
					"patching_rect": [
						30.0,
						290.0,
						400.0,
						22.0
					],
					"text": "jit.gl.pix sr_corruption_ctx @gen sr.corruption @mode 0 @intensity 0.5 @seed 42"
				}
			},
			{
				"box": {
					"id": "obj-16",
					"maxclass": "jit.pwindow",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"jit_matrix",
						""
					],
					"patching_rect": [
						30.0,
						360.0,
						320.0,
						180.0
					]
				}
			},
			{
				"box": {
					"id": "obj-17",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						400.0,
						460.0,
						350.0,
						120.0
					],
					"text": "Corruption Modes:\n\nXOR (0): Simulates bitwise XOR with random noise\nInvert (1): Blends with inverted colors\nShuffle (2): Random per-pixel channel permutation\n\nRequires tile mask on second input from sr.tilegen."
				}
			}
		],
		"lines": [
			{
				"patchline": {
					"source": [
						"obj-mode-btn",
						0
					],
					"destination": [
						"obj-mode-counter",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-mode-counter",
						0
					],
					"destination": [
						"obj-mode-select",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-mode-select",
						0
					],
					"destination": [
						"obj-m0",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-mode-select",
						1
					],
					"destination": [
						"obj-m1",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-mode-select",
						2
					],
					"destination": [
						"obj-m2",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-mode-select",
						0
					],
					"destination": [
						"obj-n0",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-mode-select",
						1
					],
					"destination": [
						"obj-n1",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-mode-select",
						2
					],
					"destination": [
						"obj-n2",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-m0",
						0
					],
					"destination": [
						"obj-15",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-m1",
						0
					],
					"destination": [
						"obj-15",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-m2",
						0
					],
					"destination": [
						"obj-15",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-n0",
						0
					],
					"destination": [
						"obj-mode-name",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-n1",
						0
					],
					"destination": [
						"obj-mode-name",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-n2",
						0
					],
					"destination": [
						"obj-mode-name",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-dial-intensity",
						0
					],
					"destination": [
						"obj-10",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-dial-seed",
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
						"obj-20",
						0
					],
					"destination": [
						"obj-world",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-20",
						0
					],
					"destination": [
						"obj-delay",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-delay",
						0
					],
					"destination": [
						"obj-21",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-21",
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
						"obj-3",
						0
					],
					"destination": [
						"obj-4",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-4",
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
						"obj-4",
						0
					],
					"destination": [
						"obj-noise",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-noise",
						0
					],
					"destination": [
						"obj-op",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-op",
						0
					],
					"destination": [
						"obj-tex",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-tex",
						0
					],
					"destination": [
						"obj-15",
						1
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
						"obj-15",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-7",
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
						"obj-15",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-10",
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
						"obj-15",
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
			},
			{
				"patchline": {
					"source": [
						"obj-14",
						0
					],
					"destination": [
						"obj-15",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-15",
						0
					],
					"destination": [
						"obj-16",
						0
					]
				}
			}
		]
	}
}