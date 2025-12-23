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
			650.0
		],
		"description": "Landsat 7 SLC-off wedge artifact simulation",
		"digest": "Simulates scan line corrector failure with wedge-shaped gaps",
		"tags": "jitter, GPU, satellite, landsat, slc-off, effect",
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
					"text": "sr.slcoff + sr.maskgen - SLC-off Wedge Artifact Simulation",
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
						60.0
					],
					"text": "Simulates Landsat 7 Scan Line Corrector failure.\nsr.maskgen (CPU): Generates wedge-shaped gap mask\nsr.slcoff (GPU): Applies fill based on mask\n\nGaps widen toward image edges, creating characteristic wedge pattern."
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
					"text": "jit.world sr_slcoff_ctx @visible 0"
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
					"text": "jit.movie @autostart 1 @loop 1 @output_texture 1 @drawto sr_slcoff_ctx"
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
						120.0,
						150.0,
						20.0
					],
					"text": "gap_width: 0.0-0.5"
				}
			},
			{
				"box": {
					"id": "obj-7",
					"maxclass": "flonum",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						450.0,
						145.0,
						60.0,
						22.0
					],
					"minimum": 0.0,
					"maximum": 0.5
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
						175.0,
						100.0,
						22.0
					],
					"text": "gap_width $1"
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
						120.0,
						100.0,
						20.0
					],
					"text": "scan_period: 2-100"
				}
			},
			{
				"box": {
					"id": "obj-10",
					"maxclass": "number",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						570.0,
						160.0,
						50.0,
						22.0
					],
					"minimum": 2,
					"maximum": 100
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
						570.0,
						190.0,
						100.0,
						22.0
					],
					"text": "scan_period $1"
				}
			},
			{
				"box": {
					"id": "obj-12",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						250.0,
						220.0,
						200.0,
						20.0
					],
					"text": "CPU: Generate wedge mask"
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
						240.0,
						200.0,
						22.0
					],
					"text": "sr.maskgen @gap_width 0.22 @scan_period 16"
				}
			},
			{
				"box": {
					"id": "obj-14",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						450.0,
						220.0,
						200.0,
						40.0
					],
					"text": "fill_mode:\n0=black, 1=white, 2=mean neighbor"
				}
			},
			{
				"box": {
					"id": "obj-fill-btn",
					"maxclass": "button",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"bang"
					],
					"patching_rect": [
						560.0,
						270.0,
						24.0,
						24.0
					]
				}
			},
			{
				"box": {
					"id": "obj-fill-counter",
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
						590.0,
						272.0,
						70.0,
						22.0
					],
					"text": "counter 0 2"
				}
			},
			{
				"box": {
					"id": "obj-fill-select",
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
						665.0,
						272.0,
						70.0,
						22.0
					],
					"text": "select 0 1 2"
				}
			},
			{
				"box": {
					"id": "obj-fill-name",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						740.0,
						272.0,
						50.0,
						20.0
					],
					"text": "Black"
				}
			},
			{
				"box": {
					"id": "obj-fm0",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						560.0,
						310.0,
						70.0,
						22.0
					],
					"text": "fill_mode 0"
				}
			},
			{
				"box": {
					"id": "obj-fm1",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						635.0,
						310.0,
						70.0,
						22.0
					],
					"text": "fill_mode 1"
				}
			},
			{
				"box": {
					"id": "obj-fm2",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						710.0,
						310.0,
						70.0,
						22.0
					],
					"text": "fill_mode 2"
				}
			},
			{
				"box": {
					"id": "obj-fname0",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						560.0,
						340.0,
						55.0,
						22.0
					],
					"text": "set Black"
				}
			},
			{
				"box": {
					"id": "obj-fname1",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						620.0,
						340.0,
						55.0,
						22.0
					],
					"text": "set White"
				}
			},
			{
				"box": {
					"id": "obj-fname2",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						680.0,
						340.0,
						55.0,
						22.0
					],
					"text": "set Mean"
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
						450.0,
						270.0,
						50.0,
						22.0
					],
					"minimum": 0,
					"maximum": 2
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
						450.0,
						300.0,
						100.0,
						22.0
					],
					"text": "fill_mode $1"
				}
			},
			{
				"box": {
					"id": "obj-17",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						250.0,
						320.0,
						200.0,
						20.0
					],
					"text": "GPU: Apply fill using mask"
				}
			},
			{
				"box": {
					"id": "obj-18",
					"maxclass": "newobj",
					"numinlets": 2,
					"numoutlets": 2,
					"outlettype": [
						"jit_gl_texture",
						""
					],
					"patching_rect": [
						30.0,
						340.0,
						400.0,
						22.0
					],
					"text": "jit.gl.pix sr_slcoff_ctx @gen sr.slcoff @fill_mode 0"
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
						400.0,
						320.0,
						180.0
					]
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
						"obj-4",
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
						"obj-5",
						0
					],
					"destination": [
						"obj-18",
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
						"obj-13",
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
						"obj-18",
						1
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
						"obj-18",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-18",
						0
					],
					"destination": [
						"obj-19",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fill-btn",
						0
					],
					"destination": [
						"obj-fill-counter",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fill-counter",
						0
					],
					"destination": [
						"obj-fill-select",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fill-select",
						0
					],
					"destination": [
						"obj-fm0",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fill-select",
						1
					],
					"destination": [
						"obj-fm1",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fill-select",
						2
					],
					"destination": [
						"obj-fm2",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fill-select",
						0
					],
					"destination": [
						"obj-fname0",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fill-select",
						1
					],
					"destination": [
						"obj-fname1",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fill-select",
						2
					],
					"destination": [
						"obj-fname2",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fm0",
						0
					],
					"destination": [
						"obj-18",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fm1",
						0
					],
					"destination": [
						"obj-18",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fm2",
						0
					],
					"destination": [
						"obj-18",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fname0",
						0
					],
					"destination": [
						"obj-fill-name",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fname1",
						0
					],
					"destination": [
						"obj-fill-name",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-fname2",
						0
					],
					"destination": [
						"obj-fill-name",
						0
					]
				}
			}
		]
	}
}