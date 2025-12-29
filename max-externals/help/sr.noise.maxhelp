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
			680.0
		],
		"bglocked": 0,
		"openinpresentation": 0,
		"default_fontsize": 12.0,
		"default_fontface": 0,
		"default_fontname": "Arial",
		"gridonopen": 1,
		"gridsize": [
			15.0,
			15.0
		],
		"gridsnaponopen": 1,
		"objectsnaponopen": 1,
		"statusbarvisible": 2,
		"toolbarvisible": 1,
		"lefttoolbarpinned": 0,
		"toptoolbarpinned": 0,
		"righttoolbarpinned": 0,
		"bottomtoolbarpinned": 0,
		"toolbars_unpinned_last_save": 0,
		"tallnewobj": 0,
		"boxanimatetime": 200,
		"enablehscroll": 1,
		"enablevscroll": 1,
		"devicewidth": 0.0,
		"description": "Noise effect with Gaussian, row, and column modes",
		"digest": "Add noise to images using PCG-based random generation",
		"tags": "jitter, GPU, noise, glitch, effect",
		"style": "",
		"subpatcher_template": "",
		"assistshowspatchername": 0,
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
					"text": "sr.noise - Noise Effect Shader for SevenRad",
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
						40.0
					],
					"text": "Adds noise to images using three different modes:\n- Gaussian (mode 0): Per-pixel random noise\n- Row (mode 1): Horizontal scanline artifacts\n- Column (mode 2): Vertical striping artifacts"
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
					"text": "jit.world sr_noise_ctx @visible 0"
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
						120.0,
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
						150.0,
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
						180.0,
						380.0,
						22.0
					],
					"text": "jit.movie @autostart 1 @loop 1 @output_texture 1 @drawto sr_noise_ctx"
				}
			},
			{
				"box": {
					"id": "obj-21",
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
						120.0,
						63.0,
						22.0
					],
					"text": "delay 100"
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
						250.0,
						150.0,
						120.0,
						22.0
					],
					"text": "read chickens.mp4"
				}
			},
			{
				"box": {
					"id": "obj-7",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						450.0,
						230.0,
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
						255.0,
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
						285.0,
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
						315.0,
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
						345.0,
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
						370.0,
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
						395.0,
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
						345.0,
						55.0,
						22.0
					],
					"text": "set Gauss"
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
						370.0,
						55.0,
						22.0
					],
					"text": "set Row"
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
						395.0,
						55.0,
						22.0
					],
					"text": "set Col"
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
						257.0,
						50.0,
						20.0
					],
					"text": "Gauss"
				}
			},
			{
				"box": {
					"id": "obj-8",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						450.0,
						420.0,
						200.0,
						20.0
					],
					"text": "manual mode (0-2)"
				}
			},
			{
				"box": {
					"id": "obj-9",
					"maxclass": "number",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						450.0,
						445.0,
						50.0,
						22.0
					],
					"minimum": 0,
					"maximum": 2
				}
			},
			{
				"box": {
					"id": "obj-10",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						450.0,
						475.0,
						80.0,
						22.0
					],
					"text": "mode $1"
				}
			},
			{
				"box": {
					"id": "obj-11",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						580.0,
						230.0,
						100.0,
						20.0
					],
					"text": "amount: 0.0-1.0"
				}
			},
			{
				"box": {
					"id": "obj-dial-amount",
					"maxclass": "dial",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"float"
					],
					"patching_rect": [
						580.0,
						255.0,
						40.0,
						40.0
					],
					"size": 100.0,
					"min": 0.0,
					"mult": 0.01,
					"floatoutput": 1
				}
			},
			{
				"box": {
					"id": "obj-12",
					"maxclass": "flonum",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						630.0,
						270.0,
						60.0,
						22.0
					],
					"minimum": 0.0,
					"maximum": 1.0,
					"numdecimalplaces": 3
				}
			},
			{
				"box": {
					"id": "obj-13",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						630.0,
						300.0,
						80.0,
						22.0
					],
					"text": "amount $1"
				}
			},
			{
				"box": {
					"id": "obj-14",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						710.0,
						230.0,
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
						710.0,
						255.0,
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
					"id": "obj-15",
					"maxclass": "number",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						760.0,
						270.0,
						50.0,
						22.0
					],
					"minimum": 0,
					"maximum": 1000
				}
			},
			{
				"box": {
					"id": "obj-16",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						760.0,
						300.0,
						60.0,
						22.0
					],
					"text": "seed $1"
				}
			},
			{
				"box": {
					"id": "obj-17",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"jit_gl_texture",
						""
					],
					"patching_rect": [
						30.0,
						250.0,
						400.0,
						22.0
					],
					"text": "jit.gl.pix sr_noise_ctx @gen sr.noise @mode 0 @amount 0.1 @seed 42"
				}
			},
			{
				"box": {
					"id": "obj-18",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						30.0,
						320.0,
						180.0,
						22.0
					],
					"text": "jit.gl.render @erase_color 0 0 0 1"
				}
			},
			{
				"box": {
					"id": "obj-19",
					"maxclass": "jit.pwindow",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"jit_matrix",
						""
					],
					"patching_rect": [
						30.0,
						380.0,
						320.0,
						180.0
					]
				}
			},
			{
				"box": {
					"id": "obj-20",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						400.0,
						510.0,
						350.0,
						100.0
					],
					"text": "PCG Hash Algorithm:\nUses Permuted Congruential Generator for high-quality deterministic random numbers. Same seed produces identical results across sessions.\n\nConstants match Python/Taichi implementation for validation."
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
						"obj-17",
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
						"obj-17",
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
						"obj-17",
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
						"obj-dial-amount",
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
						"obj-dial-seed",
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
						"obj-21",
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
						"obj-21",
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
						"obj-6",
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
						"obj-5",
						0
					],
					"destination": [
						"obj-17",
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
						"obj-5",
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
						"obj-10",
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
						"obj-17",
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
						"obj-17",
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
			},
			{
				"patchline": {
					"source": [
						"obj-16",
						0
					],
					"destination": [
						"obj-17",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-17",
						0
					],
					"destination": [
						"obj-19",
						0
					]
				}
			}
		]
	}
}