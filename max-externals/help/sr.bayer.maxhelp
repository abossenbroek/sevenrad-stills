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
			750.0
		],
		"description": "Bayer filter mosaicing and demosaicing for sensor simulation",
		"digest": "Two-pass Bayer CFA filter effect",
		"tags": "jitter, GPU, bayer, sensor, demosaic, effect",
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
					"text": "sr.bayer - Bayer Filter Mosaic/Demosaic (Two-Pass)",
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
					"text": "Simulates digital camera sensor Bayer Color Filter Array.\nPass 1 (sr.bayer.mosaic): Extracts single channel per pixel based on pattern.\nPass 2 (sr.bayer.demosaic): Reconstructs RGB using bilinear interpolation.\nCreates characteristic demosaicing artifacts (zipper, false color)."
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
					"text": "jit.world sr_bayer_ctx @visible 0"
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
					"text": "jit.movie @autostart 1 @loop 1 @output_texture 1 @drawto sr_bayer_ctx"
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
						150.0,
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
						180.0,
						120.0,
						22.0
					],
					"text": "read chickens.mp4"
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
						160.0,
						300.0,
						80.0
					],
					"text": "Bayer Patterns (2x2 blocks):\n0 RGGB: R G    1 BGGR: B G\n        G B            G R\n2 GRBG: G R    3 GBRG: G B\n        B G            R G"
				}
			},
			{
				"box": {
					"id": "obj-cycle-label",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						450.0,
						250.0,
						120.0,
						20.0
					],
					"text": "Click to cycle pattern:"
				}
			},
			{
				"box": {
					"id": "obj-button",
					"maxclass": "button",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"bang"
					],
					"patching_rect": [
						450.0,
						275.0,
						24.0,
						24.0
					]
				}
			},
			{
				"box": {
					"id": "obj-counter",
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
						310.0,
						73.0,
						22.0
					],
					"text": "counter 0 3"
				}
			},
			{
				"box": {
					"id": "obj-pattern-num",
					"maxclass": "number",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						480.0,
						275.0,
						50.0,
						22.0
					],
					"triangle": 0
				}
			},
			{
				"box": {
					"id": "obj-prepend",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						450.0,
						345.0,
						95.0,
						22.0
					],
					"text": "prepend pattern"
				}
			},
			{
				"box": {
					"id": "obj-select",
					"maxclass": "newobj",
					"numinlets": 5,
					"numoutlets": 5,
					"outlettype": [
						"bang",
						"bang",
						"bang",
						"bang",
						""
					],
					"patching_rect": [
						560.0,
						310.0,
						85.0,
						22.0
					],
					"text": "select 0 1 2 3"
				}
			},
			{
				"box": {
					"id": "obj-name-0",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						560.0,
						345.0,
						45.0,
						22.0
					],
					"text": "RGGB"
				}
			},
			{
				"box": {
					"id": "obj-name-1",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						610.0,
						345.0,
						45.0,
						22.0
					],
					"text": "BGGR"
				}
			},
			{
				"box": {
					"id": "obj-name-2",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						660.0,
						345.0,
						45.0,
						22.0
					],
					"text": "GRBG"
				}
			},
			{
				"box": {
					"id": "obj-name-3",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						710.0,
						345.0,
						45.0,
						22.0
					],
					"text": "GBRG"
				}
			},
			{
				"box": {
					"id": "obj-display",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						560.0,
						380.0,
						80.0,
						22.0
					],
					"text": "RGGB"
				}
			},
			{
				"box": {
					"id": "obj-display-label",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						645.0,
						380.0,
						120.0,
						20.0
					],
					"text": "<- Current pattern"
				}
			},
			{
				"box": {
					"id": "obj-scale-label",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						450.0,
						415.0,
						180.0,
						20.0
					],
					"text": "Scale (1=subtle, 20=extreme):"
				}
			},
			{
				"box": {
					"id": "obj-scale-dial",
					"maxclass": "dial",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"float"
					],
					"patching_rect": [
						450.0,
						440.0,
						40.0,
						40.0
					],
					"size": 19.0,
					"min": 1.0,
					"mult": 1.0,
					"floatoutput": 1
				}
			},
			{
				"box": {
					"id": "obj-scale-num",
					"maxclass": "number",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						500.0,
						455.0,
						50.0,
						22.0
					],
					"minimum": 1,
					"maximum": 20
				}
			},
			{
				"box": {
					"id": "obj-prepend-scale",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						450.0,
						490.0,
						85.0,
						22.0
					],
					"text": "prepend scale"
				}
			},
			{
				"box": {
					"id": "obj-9",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						250.0,
						220.0,
						180.0,
						20.0
					],
					"text": "Pass 1: Mosaic (CFA simulation)"
				}
			},
			{
				"box": {
					"id": "obj-10",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"jit_gl_texture",
						""
					],
					"patching_rect": [
						30.0,
						240.0,
						400.0,
						22.0
					],
					"text": "jit.gl.pix sr_bayer_ctx @gen sr.bayer.mosaic @pattern 0 @scale 1"
				}
			},
			{
				"box": {
					"id": "obj-11",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						250.0,
						280.0,
						200.0,
						20.0
					],
					"text": "Pass 2: Demosaic (reconstruction)"
				}
			},
			{
				"box": {
					"id": "obj-12",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"jit_gl_texture",
						""
					],
					"patching_rect": [
						30.0,
						300.0,
						400.0,
						22.0
					],
					"text": "jit.gl.pix sr_bayer_ctx @gen sr.bayer.demosaic @pattern 0 @scale 1"
				}
			},
			{
				"box": {
					"id": "obj-13",
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
					"id": "obj-14",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						560.0,
						420.0,
						250.0,
						100.0
					],
					"text": "Demosaicing uses bilinear interpolation:\n- R pixels: G from cross, B from diagonal\n- G pixels: R/B from neighbors\n- B pixels: G from cross, R from diagonal\n\nBoth passes must use same pattern and scale!"
				}
			},
			{
				"box": {
					"id": "obj-note",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						30.0,
						560.0,
						400.0,
						40.0
					],
					"text": "Note: The @pattern/@scale text in jit.gl.pix won't update at runtime.\nUse the controls on the right to change values.",
					"fontsize": 10.0
				}
			}
		],
		"lines": [
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
						"obj-5",
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
						"obj-button",
						0
					],
					"destination": [
						"obj-counter",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-counter",
						0
					],
					"destination": [
						"obj-pattern-num",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-counter",
						0
					],
					"destination": [
						"obj-prepend",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-counter",
						0
					],
					"destination": [
						"obj-select",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-prepend",
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
						"obj-prepend",
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
						"obj-select",
						0
					],
					"destination": [
						"obj-name-0",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-select",
						1
					],
					"destination": [
						"obj-name-1",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-select",
						2
					],
					"destination": [
						"obj-name-2",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-select",
						3
					],
					"destination": [
						"obj-name-3",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-name-0",
						0
					],
					"destination": [
						"obj-display",
						1
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-name-1",
						0
					],
					"destination": [
						"obj-display",
						1
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-name-2",
						0
					],
					"destination": [
						"obj-display",
						1
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-name-3",
						0
					],
					"destination": [
						"obj-display",
						1
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-scale-dial",
						0
					],
					"destination": [
						"obj-scale-num",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-scale-num",
						0
					],
					"destination": [
						"obj-prepend-scale",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-prepend-scale",
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
						"obj-prepend-scale",
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
						"obj-10",
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
			}
		]
	}
}
